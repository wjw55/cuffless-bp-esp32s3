"""Readiness, shared extraction, causal replay and collector lifecycle regression tests."""
import copy
import hashlib
import io
import json
import math
import sys
import threading
import unittest
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bp_precheck as bp
import collect_ppg as collector
from line_transport import LineTransportDisconnected, TransportMetadata

ROOT = Path(__file__).resolve().parents[2]


def telemetry(state, now, sensor_ms=None):
    timestamp = int(now * 1000) if sensor_ms is None else sensor_ms
    state.ingest('# stats ovf=0 i2c_errors=0 overflow_recoveries=0', now)
    state.ingest('# imu_stats fifo_overflows=0 i2c_errors=0', now)
    state.ingest(f'# motion timestamp_ms={timestamp} status=still activity_g=0.01', now)


def synthetic(seconds=30):
    t = np.arange(int(seconds*100)+1) / 100
    ir = 100000 + 2000*np.sin(2*np.pi*1.2*t)
    red = 80000 + 1000*np.sin(2*np.pi*1.2*t)
    ppg = tuple((n, int(round(s*1000)), int(r), int(v)) for n, (s,r,v) in enumerate(zip(t,red,ir)))
    imu = tuple((n, int(round(s*1000)),0,0,256) for n,s in enumerate(t))
    meta = {name:0 for name in bp.FEASIBILITY_HEALTH_COUNTERS}
    meta.update(ppg_profile='upper_arm_experimental',
                firmware_motion_updates=[{'timestamp_ms':0,'status':'still'}])
    return bp.Snapshot(0, 30, seconds, ppg, imu, meta)


def populated():
    state = bp.PrecheckState()
    snapshot = synthetic()
    state.ppg.extend(snapshot.ppg);state.imu.extend(snapshot.imu)
    state.seen.update(ppg=30,imu=30)
    telemetry(state,30)
    state.previous.update(ppg=snapshot.ppg[-1],imu=snapshot.imu[-1])
    return state


class ImmediateExecutor:
    """Run worker tasks deterministically; production uses a real thread pool."""
    def __init__(self, **kwargs):pass
    def submit(self, fn, *args):
        result=Future()
        try:result.set_result(fn(*args))
        except Exception as exc:result.set_exception(exc)
        return result
    def shutdown(self, **kwargs):pass


def passing(snapshot, settings):
    return bp.Assessment(snapshot,True,6,6,28.0,'')


class Clock:
    def __init__(self):self.value=0.0
    def __call__(self):return self.value


class Source:
    def __init__(self, clock, transport='serial', limit=5000):
        self.clock=clock;self.transport=transport;self.lines=self.generate(limit)
        self.read=[];self.opens=self.closes=self.resets=0
        self.metadata=TransportMetadata(transport,'fake-device')
        self.display_name='fake-device'
    def __enter__(self):self.opens+=1;return self
    def __exit__(self,*args):self.closes+=1
    def set_buffer_size(self,**kwargs):pass
    def reset_input_buffer(self):self.resets+=1
    @staticmethod
    def generate(limit):
        for n in range(limit):
            if n%100==0:
                yield '# stats ovf=0 i2c_errors=0 overflow_recoveries=0'
                yield '# imu_stats fifo_overflows=0 i2c_errors=0'
                yield f'# motion timestamp_ms={n*10} status=still activity_g=0.01'
            yield f'{n},{n*10},80000,100000'
            yield f'imu,{n},{n*10},0,0,256'
    def readline(self):
        self.clock.value+=0.005
        s=next(self.lines,'');self.read.append(s)
        return s.encode()


class ReadinessStateTests(unittest.TestCase):
    def test_fresh_complete_still_window_can_be_ready(self):
        state=populated();snap=state.snapshot(30)
        state.result=passing(snap,None)
        self.assertTrue(state.ready(30));self.assertEqual(state.duration(),30)

    def test_full_30_seconds_required_even_when_windows_pass(self):
        state=populated();state.result=passing(bp.Snapshot(0,30,29.99,(),(),{}),None)
        self.assertFalse(state.ready(30))

    def test_both_streams_and_health_are_required(self):
        for field,key in [('seen','imu'),('health_seen','stats'),('health_seen','imu_stats')]:
            with self.subTest(field=field,key=key):
                state=populated();state.result=passing(state.snapshot(30),None)
                getattr(state,field).pop(key)
                self.assertFalse(state.ready(30))

    def test_stale_data_motion_health_hide_ready_and_reset(self):
        for field,key in [('seen','ppg'),('seen','imu'),('health_seen','stats'),('health_seen','imu_stats')]:
            state=populated();state.result=passing(state.snapshot(30),None)
            getattr(state,field)[key]=0
            self.assertFalse(state.ready(30));self.assertEqual(state.duration(),0)
        state=populated();state.motion_seen=0
        self.assertFalse(state.ready(30));self.assertEqual(state.duration(),0)

    def test_missing_motion_and_malformed_status_never_pass(self):
        state=populated();state.motion_seen=None
        self.assertFalse(state.ready(30))
        state=populated();state.ingest('# motion status=still timestamp_ms=oops',30)
        self.assertFalse(state.ready(30));self.assertEqual(state.motion_status,'unknown')

    def test_missing_counter_is_not_assumed_zero(self):
        state=populated();state.ingest('# stats ovf=0',30)
        self.assertNotIn('stats',state.health_seen)
        self.assertFalse(state.ready(30))

    def test_health_freshness_allows_five_second_reporting_jitter(self):
        state=populated()
        state.seen.update(ppg=35.2,imu=35.2)
        state.ingest('# motion timestamp_ms=35200 status=still',35.2)
        self.assertIsNone(state.guards(35.2))
        self.assertEqual(state.duration(),30)
        state.seen.update(ppg=40.1,imu=40.1)
        state.ingest('# motion timestamp_ms=40100 status=still',40.1)
        self.assertIn('health',state.guards(40.1))
        self.assertEqual(state.duration(),0)

    def test_all_reported_sensor_faults_are_fatal(self):
        lines=['# stats ovf=1 i2c_errors=0', '# stats ovf=0 i2c_errors=1',
               '# imu_stats fifo_overflows=1 i2c_errors=0', '# imu_stats fifo_overflows=0 i2c_errors=1',
               '# stats ovf=0 i2c_errors=0 overflow_recoveries=1',
               '# warning event=fifo_overflow', '# warning event=imu_fifo_overflow',
               '# warning event=fifo_overflow_recovery_failed', '# warning event=imu_not_detected',
               '# warning event=imu_init_failed',
               '# warning event=fifo_pointer_read_failed i2c_errors=2',
               '# stats ovf=0 i2c_errors=nan', '# imu_stats fifo_overflows=-1 i2c_errors=0']
        for line in lines:
            with self.subTest(line=line),self.assertRaises(bp.PrecheckError):
                populated().ingest(line,30)

    def test_movement_and_old_worker_generation_cannot_start(self):
        state=populated();old=passing(state.snapshot(30),None)
        state.ingest('# motion timestamp_ms=30000 status=moving',30)
        telemetry(state,31);state.result=old
        self.assertFalse(state.ready(31));self.assertEqual(state.duration(),0)
        self.assertGreater(state.generation,old.snapshot.generation)

    def test_sequence_restart_gaps_and_timing_fault_reset_both_streams(self):
        for line in ['3002,30010,80000,100000','1,5,80000,100000',
                     '3001,30000,80000,100000','3001,30100,80000,100000',
                     'imu,3002,30010,0,0,256','imu,3001,30100,0,0,256']:
            with self.subTest(line=line):
                state=populated();state.ingest(line,30)
                self.assertGreater(state.generation,0);self.assertEqual(state.duration(),0)

    def test_delayed_same_generation_result_is_not_ready(self):
        state=populated();state.result=passing(state.snapshot(30),None)
        telemetry(state,34);state.seen.update(ppg=34,imu=34)
        self.assertFalse(state.ready(34))

    def test_timestamps_align_unequal_rates_without_extrapolation(self):
        state=populated()
        state.imu=deque(((n,int(t),0,0,256) for n,t in enumerate(range(500,31001,8))),maxlen=bp.MAX_SAMPLES)
        snap=state.snapshot(30)
        self.assertEqual(snap.duration_s,29.5)
        self.assertGreaterEqual(min(r[1] for r in snap.ppg),500)
        self.assertLessEqual(max(r[1] for r in snap.imu),30000)

    def test_buffer_and_history_are_bounded_and_bad_lines_ignored(self):
        state=bp.PrecheckState()
        for n in range(10000):
            now=n/100
            if n%100==0:telemetry(state,now)
            state.ingest(f'{n},{n*10},80000,100000',now)
            state.ingest(f'imu,{n},{n*10},0,0,256',now)
        self.assertLessEqual(len(state.ppg),3501);self.assertLessEqual(len(state.imu),3501)
        generation=state.generation
        for line in ['nonsense','sample_seq,timestamp_ms,red,ir','-1,x,y,z']:
            state.ingest(line,100)
        self.assertEqual(state.generation,generation)
        snap=state.snapshot(99.99)
        self.assertLessEqual(snap.duration_s,30)


class SharedExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.settings=bp.load_settings(ROOT)

    def test_real_clean_signal_uses_shared_rules_and_unique_coverage(self):
        before=copy.deepcopy(self.settings.config)
        snap=synthetic();result=bp.assess(snap,self.settings)
        self.assertTrue(result.usable);self.assertEqual(result.total,6)
        self.assertGreaterEqual(result.coverage_s,24)
        self.assertLess(result.coverage_s,result.accepted*8)
        self.assertEqual(self.settings.config,before)

    def test_contact_clipping_and_inconsistent_waveform_rejected(self):
        for kind in ['contact','clipping','flat','step','inconsistent']:
            with self.subTest(kind=kind):
                snap=synthetic();rows=[]
                noise=np.random.default_rng(42).normal(0,2000,len(snap.ppg))
                for seq,t,r,ir in snap.ppg:
                    if kind=='contact':ir=20000
                    elif kind=='clipping':ir=262143
                    elif kind=='flat':ir=100000
                    elif kind=='step':ir+=40000*(int(t/4000)%2)
                    elif kind=='inconsistent':ir=int(100000+noise[seq])
                    rows.append((seq,t,r,ir))
                changed=bp.Snapshot(0,30,30,tuple(rows),snap.imu,snap.metadata)
                result=bp.assess(changed,self.settings)
                self.assertFalse(result.usable)
                self.assertTrue(result.reason)

    def test_shared_missing_samples_motion_and_health_reject(self):
        for kind in ['missing','motion','health']:
            snap=synthetic();rows=list(snap.ppg);meta=copy.deepcopy(snap.metadata)
            if kind=='missing':rows.pop(1000)
            if kind=='motion':meta['firmware_motion_updates']=[{'timestamp_ms':0,'status':'moving'}]
            if kind=='health':meta['imu_firmware_i2c_error_count']=1
            result=bp.assess(bp.Snapshot(0,30,30,tuple(rows),snap.imu,meta),self.settings)
            self.assertFalse(result.usable)

    def test_no_bp_prediction_or_cuff_labels_used(self):
        snap=synthetic();snap.metadata.update(systolic_mmHg=999,diastolic_mmHg=1,cuff_hr_bpm=999)
        with patch('bp_core.inference.predict_frame',side_effect=AssertionError('No BP inference')):
            first=bp.assess(snap,self.settings)
            snap.metadata.update(systolic_mmHg=90,diastolic_mmHg=60,cuff_hr_bpm=50)
            second=bp.assess(snap,self.settings)
        self.assertEqual((first.usable,first.accepted,first.coverage_s),(second.usable,second.accepted,second.coverage_s))

    def test_local_008_and_009_causal_trailing_windows_without_raw_mutation(self):
        for trial in ['008','009']:
            prefix=ROOT/'data/raw'/('P001_housing_v1_prospective_test_001_validation_'+trial)
            paths=[Path(str(prefix)+s) for s in ['_metadata.json','_ppg.csv','_imu.csv']]
            if not all(p.exists() for p in paths):
                self.skipTest('Local participant fixtures are deliberately not committed')
            hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            meta=json.loads(paths[0].read_text());ppg=pd.read_csv(paths[1]);imu=pd.read_csv(paths[2])
            base=max(ppg.timestamp_ms.iloc[0],imu.timestamp_ms.iloc[0]);stop=min(ppg.timestamp_ms.iloc[-1],imu.timestamp_ms.iloc[-1])
            results=[]
            for end in range(int(base)+30000,int(stop)+1,5000):
                a=ppg[(ppg.timestamp_ms>=end-30000)&(ppg.timestamp_ms<=end)]
                b=imu[(imu.timestamp_ms>=end-30000)&(imu.timestamp_ms<=end)]
                metadata={k:meta.get(k,0) for k in bp.FEASIBILITY_HEALTH_COUNTERS}
                metadata['ppg_profile']='upper_arm_experimental'
                metadata['firmware_motion_updates']=[x for x in meta['firmware_motion_updates'] if x['timestamp_ms']<=end]
                snapshot=bp.Snapshot(0,0,30,tuple(a.itertuples(index=False,name=None)),tuple(b.itertuples(index=False,name=None)),metadata)
                results.append((int(round((end-base)/1000)),bp.assess(snapshot,self.settings)))
            if trial=='008':self.assertFalse(any(r.usable for _,r in results))
            else:
                self.assertFalse(results[0][1].usable)
                at,result=next((at,r) for at,r in results if r.usable)
                self.assertEqual(at,35);self.assertEqual(result.accepted,5);self.assertEqual(result.coverage_s,28)
            self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])


class RunnerTests(unittest.TestCase):
    def setUp(self):self.settings=bp.load_settings(ROOT)

    def test_success_uses_same_source_without_reset_and_saves_provenance(self):
        clock=Clock();source=Source(clock);out=io.StringIO()
        with patch.object(bp,'ThreadPoolExecutor',ImmediateExecutor):
            result=bp.run_precheck(source,self.settings,clock=clock,output=lambda s,**kw:print(s,file=out),evaluator=passing)
        self.assertGreaterEqual(clock.value,30);self.assertEqual(source.resets,0)
        self.assertEqual(result.metadata['status'],'passed')
        self.assertEqual(result.metadata['config_sha256'],self.settings.provenance['config_sha256'])
        self.assertEqual(result.diagnostics['hr_updates'],[])
        self.assertLessEqual(len(result.diagnostics['motion_updates']),1)
        self.assertIn('READY — recording starts now',out.getvalue())

    def test_timeout_and_disconnect_propagate(self):
        clock=Clock();source=Source(clock)
        with patch.object(bp,'ThreadPoolExecutor',ImmediateExecutor),self.assertRaisesRegex(bp.PrecheckError,'timed out'):
            bp.run_precheck(source,self.settings,.02,clock=clock,output=lambda *a,**k:None,evaluator=passing)
        source.readline=lambda: (_ for _ in ()).throw(LineTransportDisconnected('lost BLE'))
        with self.assertRaises(LineTransportDisconnected):
            bp.run_precheck(source,self.settings,clock=clock,output=lambda *a,**k:None)

    def test_worker_does_not_block_reading_or_accumulate_tasks(self):
        clock=Clock();source=Source(clock);entered=threading.Event();release=threading.Event();calls=[]
        def slow(snapshot,settings):
            calls.append(snapshot);entered.set();release.wait(5)
            return passing(snapshot,settings)
        real_read=source.readline
        def read():
            if calls:entered.wait(.1)
            return real_read()
        source.readline=read
        try:
            with self.assertRaises(bp.PrecheckError):
                bp.run_precheck(source,self.settings,15,clock=clock,output=lambda *a,**k:None,evaluator=slow)
            self.assertTrue(entered.wait(1));self.assertEqual(len(calls),1)
            self.assertGreater(len(source.read),2000)
        finally:release.set()


class CollectorLifecycleTests(unittest.TestCase):
    def test_cli_is_opt_in_and_incompatible_options_fail(self):
        base=['--port','FAKE','--duration','90','--subject','TEST','--session','test']
        args=collector.parse_args(base)
        self.assertFalse(args.precheck_bp_quality);self.assertEqual(args.precheck_timeout,120)
        cases=[[],['--ppg-profile','upper_arm_experimental','--precheck-timeout','nan'],
               ['--ppg-profile','upper_arm_experimental','--precheck-timeout','inf'],
               ['--ppg-profile','upper_arm_experimental','--motion-protocol','motion_quality_v1']]
        for extra in cases:
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                collector.parse_args(base+['--precheck-bp-quality']+extra)

    def run_collector(self,folder,transport='serial',precheck=None,read_failure=None,overwrite=False):
        clock=Clock();source=Source(clock,transport)
        args=['collect_ppg.py','--transport',transport,'--duration','0.15','--subject','TEST',
              '--session','precheck','--trial-id','one','--ppg-profile','upper_arm_experimental',
              '--precheck-bp-quality','--outdir',str(folder),'--labels-dir',str(folder/'labels')]
        args+=['--port','FAKE'] if transport=='serial' else ['--ble-device','PPG-LOGGER-FAKE']
        if overwrite:args+=['--overwrite']
        runner=bp.run_precheck
        boundary=[]
        def perform(ser,settings,timeout):
            if precheck is not None:raise precheck
            result=runner(ser,settings,timeout,clock=clock,output=lambda *a,**k:None,evaluator=passing)
            boundary.append(len(source.read))
            if read_failure:
                source.readline=lambda: (_ for _ in ()).throw(read_failure)
            return result
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv',args))
            factory=stack.enter_context(patch.object(collector,'create_line_source',return_value=source))
            stack.enter_context(patch.object(collector.time,'monotonic',clock))
            stack.enter_context(patch.object(collector,'SERIAL_STARTUP_DELAY_S',0))
            stack.enter_context(patch.object(bp,'ThreadPoolExecutor',ImmediateExecutor))
            stack.enter_context(patch.object(bp,'run_precheck',side_effect=perform))
            for name in ['save_plot','save_zoom_plot','save_motion_plot','print_summary']:
                stack.enter_context(patch.object(collector,name))
            bp_prompt=stack.enter_context(patch.object(collector,'apply_prompt_bp_after'))
            label_prompt=stack.enter_context(patch.object(collector,'apply_prompt_labels'))
            stack.enter_context(redirect_stdout(io.StringIO()));stack.enter_context(redirect_stderr(io.StringIO()))
            code=collector.main()
        self.assertEqual(factory.call_count,1);self.assertEqual(source.opens,1);self.assertEqual(source.closes,1)
        bp_prompt.assert_not_called();label_prompt.assert_not_called()
        return code,source,boundary

    def test_serial_ble_start_only_after_precheck_and_preserve_raw_format(self):
        for transport in ['serial','ble']:
            with self.subTest(transport=transport),TemporaryDirectory() as folder:
                code,source,boundary=self.run_collector(Path(folder),transport)
                self.assertEqual(code,0)
                p=Path(folder)/'TEST_precheck_one'
                metadata=json.loads(Path(str(p)+'_metadata.json').read_text())
                frame=pd.read_csv(str(p)+'_ppg.csv');imu=pd.read_csv(str(p)+'_imu.csv')
                self.assertEqual(list(frame.columns),['sample_seq','timestamp_ms','red','ir'])
                self.assertEqual(list(imu.columns),['imu_seq','timestamp_ms','x_raw','y_raw','z_raw'])
                self.assertGreater(len(frame),5);self.assertGreater(len(imu),5)
                self.assertEqual(metadata['duration_seconds'],.15);self.assertIn('precheck',metadata)
                before=[collector.parse_ppg_row(line) for line in source.read[:boundary[0]]]
                old_ids={r[0] for r in before if r is not None}
                self.assertTrue(old_ids.isdisjoint(set(frame.sample_seq)))
                self.assertEqual(source.resets,1 if transport=='serial' else 0)

    def test_failures_before_start_leave_no_files(self):
        for error in [bp.PrecheckError('timed out'),bp.PrecheckError('Sensor fault'),KeyboardInterrupt(),LineTransportDisconnected('lost BLE')]:
            with self.subTest(error=type(error).__name__),TemporaryDirectory() as folder:
                code,_,_=self.run_collector(Path(folder),precheck=error)
                self.assertNotEqual(code,0);self.assertFalse(list(Path(folder).rglob('*')))

    def test_failed_precheck_does_not_overwrite_existing_trial(self):
        with TemporaryDirectory() as folder:
            target=Path(folder)/'TEST_precheck_one_ppg.csv'
            target.write_bytes(b'existing recording must survive')
            code,_,_=self.run_collector(Path(folder),precheck=bp.PrecheckError('timed out'),overwrite=True)
            self.assertNotEqual(code,0)
            self.assertEqual(target.read_bytes(),b'existing recording must survive')
            self.assertEqual(list(Path(folder).rglob('*')),[target])

    def test_post_start_interrupt_or_disconnect_saves_incomplete_attempt(self):
        for error in [KeyboardInterrupt(),LineTransportDisconnected('lost BLE')]:
            with TemporaryDirectory() as folder:
                code,_,_=self.run_collector(Path(folder),read_failure=error)
                self.assertEqual(code,0)
                meta=json.loads((Path(folder)/'TEST_precheck_one_metadata.json').read_text())
                self.assertTrue(meta['interrupted']);self.assertEqual(meta['precheck']['status'],'passed')


if __name__=='__main__':unittest.main()
