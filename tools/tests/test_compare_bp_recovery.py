import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_bp_recovery as recovery
from bp_core.inference import BPModelBundle, BPInferenceResult, extract_current_features
from test_bp_pipeline import minimal_config


def bundle():
    config = minimal_config(Path('.'))
    config['quality'].update(minimum_unique_clean_coverage_seconds=60,
                             require_upper_arm_analyzer_acceptance=True)
    return BPModelBundle(Path('.'), 'P001', 'cal', 120., 70., {'x':1.}, config,
                         {}, True, False, {})


def frame(seconds=40):
    t = np.arange(0, seconds + .001, .01)
    return pd.DataFrame(dict(sample_seq=np.arange(len(t)), timestamp_ms=t*1000,
                             ir=100000 + 2000*np.sin(2*np.pi*t),
                             red=70000 + 1000*np.sin(2*np.pi*t)))


class RecoveryTests(unittest.TestCase):
    def test_experimental_copy_preserves_model_and_calibration(self):
        original = bundle(); before = copy.deepcopy(original.config)
        candidate = recovery.short_bundle(original, 24)
        self.assertEqual(original.config, before)
        self.assertIs(candidate.packages, original.packages)
        self.assertIs(candidate.calibration_features, original.calibration_features)
        self.assertEqual(candidate.calibration_sbp, original.calibration_sbp)
        self.assertFalse(candidate.viewer_eligible)
        self.assertAlmostEqual(candidate.config['quality']['minimum_unique_clean_coverage_seconds'], 19.2)

    def test_trailing_slice_has_no_future_or_pre_motion_samples(self):
        selected = recovery.trailing_frame(frame(), 30000, 24, 15000)
        self.assertEqual(selected.timestamp_ms.min(), 15000)
        self.assertEqual(selected.timestamp_ms.max(), 30000)

    def test_messages_after_endpoint_cannot_change_features(self):
        meta = {'firmware_motion_updates':[{'timestamp_ms':0,'status':'still'},
                 {'timestamp_ms':15000,'status':'moving'}, {'timestamp_ms':50000,'status':'moving'}]}
        prefix = recovery.prefix_metadata(meta, 20000, 10000)
        self.assertEqual([x['timestamp_ms'] for x in prefix['firmware_motion_updates']], [0,15000])
        self.assertEqual(len(meta['firmware_motion_updates']),3)

    def test_movement_and_staleness_break_runs(self):
        messages = [{'timestamp_ms':t,'status':s} for t,s in
                    [(0,'still'),(1000,'still'),(2000,'moving'),(3000,'still'),(8000,'still')]]
        self.assertEqual(recovery.motion_runs(messages,0,12000), [[0,2000],[3000,6000],[8000,11000]])

    def test_instructed_motion_cannot_be_scored_even_if_imu_says_still(self):
        meta = {'firmware_motion_updates':[{'timestamp_ms':t,'status':'still'} for t in range(0,10001,1000)],
                'offline_still_intervals_ms':[[0,2000],[8000,10000]]}
        self.assertEqual(recovery.recording_runs(meta,0,10000),[[0,2000],[8000,10000]])

    def test_insufficient_fresh_data_never_calls_predictor(self):
        with patch.object(recovery,'predict_frame') as predictor:
            result = recovery.score(bundle(),frame(),{},'20',30000,25000)
            self.assertFalse(result['available']); predictor.assert_not_called()

    def test_contact_clipping_and_timing_still_reject(self):
        candidate = recovery.short_bundle(bundle(),20)
        for column, value in [('ir',100),('ir',262143)]:
            bad = frame(20); bad[column]=value
            occasion, _ = extract_current_features('P001',bad,{},candidate.config)
            self.assertFalse(occasion['occasion_usable'])
        bad = frame(20); bad.loc[500,'sample_seq'] += 10
        with self.assertRaises(ValueError):
            extract_current_features('P001',bad,{},candidate.config)

    def test_contact_at_trailing_edge_is_not_hidden_by_accepted_old_windows(self):
        candidate = recovery.short_bundle(bundle(),20)
        bad = frame(20)
        bad.loc[bad.index[-1],'ir'] = 100
        self.assertIn('poor_contact', recovery.current_contact_reason(bad,{},candidate.config))
        bad.loc[bad.index[-1],'ir'] = 262143
        self.assertIn('clipping', recovery.current_contact_reason(bad,{},candidate.config))

    def test_unique_coverage_not_sum_of_overlaps(self):
        from bp_core.features import aggregate_recording_features
        from bp_core.inference import _recording_context
        candidate = recovery.short_bundle(bundle(),24)
        rows = pd.DataFrame([dict(start_s=x,end_s=x+8,accepted=True,rejection_reason='',
                                  recording_id='one',feature__pulse_rate_bpm=60.) for x in [0,4,8]])
        occasion = aggregate_recording_features(_recording_context('P001'),rows,candidate.config)
        self.assertEqual(occasion['unique_clean_coverage_s'],16)
        self.assertFalse(occasion['occasion_usable'])

    def test_error_metrics_and_missing_evidence(self):
        m = recovery.error_metrics([122,116],[120,120])
        self.assertEqual(m['mae'],3); self.assertEqual(m['bias'],-1)
        self.assertEqual(m['maximum_absolute_error'],4)
        self.assertFalse(recovery.meets_criteria(recovery.error_metrics([],[]),m,1))
        self.assertFalse(recovery.meets_criteria(m,m,None))
        self.assertTrue(recovery.meets_criteria(m,m,.8))
        self.assertFalse(recovery.meets_criteria(m,m,7/9))
        worse = dict(m, mae=m['mae']+2.01)
        self.assertFalse(recovery.meets_criteria(worse,m,1))
        worse = dict(m, maximum_absolute_error=m['maximum_absolute_error']+5.01)
        self.assertFalse(recovery.meets_criteria(worse,m,1))

    def test_score_only_passes_past_and_fresh_samples_to_model(self):
        from types import SimpleNamespace
        original = frame(40)
        candidate = recovery.short_bundle(bundle(),20)
        with patch.object(recovery,'analyze_upper_arm_ppg',return_value=SimpleNamespace(status='insufficient_clean_data',status_reason='long gate')), \
             patch.object(recovery,'predict_frame',return_value=BPInferenceResult('unvalidated_estimate','experimental',120,70)) as predictor:
            result = recovery.score(candidate,original,{},'20',30000,10000)
            supplied = predictor.call_args.args[1]
            self.assertGreaterEqual(supplied.timestamp_ms.min(),10000)
            self.assertLessEqual(supplied.timestamp_ms.max(),30000)
            self.assertEqual(result['status'],'unvalidated_estimate')
            self.assertEqual(result['analyzer_status'],'insufficient_clean_data')

    def test_matched_occasions_not_repeated_updates(self):
        updates=[]; endpoints=[]
        for policy in recovery.POLICIES:
            for i in range(5):
                updates.append(dict(recording='r',end_ms=i,policy=policy,available=True))
            endpoints.append(dict(occasion_id='cuff1',policy=policy,available=True,label_eligible=True,
                                  sbp=120,dbp=70,reference_sbp=122,reference_dbp=72))
        metrics, decisions=recovery.evaluate(pd.DataFrame(updates),pd.DataFrame(endpoints),{'sbp':120,'dbp':70})
        self.assertTrue((decisions.matched_cuff_occasions==1).all())
        self.assertTrue((metrics['count']==1).all())
        with self.assertRaises(ValueError):
            recovery.evaluate(pd.DataFrame(updates),pd.DataFrame(endpoints+[endpoints[0]]),{'sbp':120,'dbp':70})


if __name__=='__main__':
    unittest.main()
