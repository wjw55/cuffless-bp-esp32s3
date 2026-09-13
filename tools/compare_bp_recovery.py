"""Offline short-window experiment; never changes a saved model or live policy.

Cuff labels enter only endpoint evaluation, after signal decisions are complete.
All replay rows are development diagnostics, not independent BP observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bp_core.datasets import _discover_local
from bp_core.features import _contact_masks, estimate_sample_rate, recording_quality_reasons
from bp_core.inference import (
    _recording_context, load_model_bundle, make_short_window_bundle, predict_frame,
    signal_from_ppg_frame,
)
from upper_arm_hr import analyze_upper_arm_ppg

DURATIONS = (20, 24, 30, 60)
POLICIES = ('baseline', *(str(x) for x in DURATIONS))
MODEL = 'data/processed/bp/20260905T235649/single_subject/P001'
PILOT = 'P001_movement_pilot_20260912_seated_001'
STALE_MS = 3000
ACTIVITIES = Path('data/processed/motion_bp_v1/seated_pilot_20260912/guarded_activity_intervals.csv')


def read_metadata(path):
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if path.stem == PILOT + '_metadata':
        blocks = pd.read_csv(ACTIVITIES)
        metadata['offline_still_intervals_ms'] = [
            [float(x.start_timestamp_ms), float(x.end_timestamp_ms)]
            for x in blocks.itertuples() if x.block in ('still_baseline', 'still_recovery')]
    return metadata


def recording_runs(metadata, first, last):
    runs = motion_runs(metadata.get('firmware_motion_updates', []), first, last)
    allowed = metadata.get('offline_still_intervals_ms')
    if allowed is not None:
        # Mild instructed movement may fall below the firmware motion threshold.
        # Admit only guarded still/recovery blocks; never publish movement BP.
        runs = [[max(a,c), min(b,d)] for a,b in runs for c,d in allowed if min(b,d)>max(a,c)]
        runs.sort()
    return runs


def short_bundle(bundle, duration):
    return make_short_window_bundle(bundle, duration, research_override=True)


def trailing_frame(frame, end_ms, duration, fresh_ms):
    time = frame.timestamp_ms.to_numpy()
    start = max(end_ms - duration * 1000, fresh_ms)
    return frame.iloc[np.searchsorted(time, start):np.searchsorted(time, end_ms, side='right')].copy()


def motion_runs(updates, first_ms, last_ms):
    """Still intervals based only on timestamped firmware messages and freshness.

    Adjacent still messages merge; moving/unknown/stale intervals break continuity.
    The last message is usable for at most STALE_MS, never indefinitely.
    """
    ordered = sorted(updates, key=lambda x: float(x['timestamp_ms']))
    runs = []
    for i, update in enumerate(ordered):
        if update.get('status') != 'still':
            continue
        t = float(update['timestamp_ms'])
        stop = min(last_ms, t + STALE_MS,
                   float(ordered[i + 1]['timestamp_ms']) if i + 1 < len(ordered) else last_ms)
        start = max(first_ms, t)
        if stop <= start:
            continue
        if runs and abs(runs[-1][1] - start) < 1e-6:
            runs[-1][1] = stop
        else:
            runs.append([start, stop])
    return runs


def prefix_metadata(metadata, end_ms, start_ms):
    result = dict(metadata)
    updates = [x for x in metadata.get('firmware_motion_updates', [])
               if float(x['timestamp_ms']) <= end_ms]
    before = [x for x in updates if float(x['timestamp_ms']) < start_ms]
    result['firmware_motion_updates'] = (before[-1:] + [x for x in updates
                                                        if float(x['timestamp_ms']) >= start_ms])
    return result


def current_contact_reason(frame, metadata, config):
    signal = signal_from_ppg_frame(frame, metadata)
    masks = _contact_masks(signal, _recording_context('P001'), estimate_sample_rate(signal.time_s), config['quality'])
    # Hard contact/clipping suppression must also cover the trailing fraction not
    # included in an eight-second feature window. Contact-step decisions remain
    # the original per-window checks (a partial final second is not a new gate).
    return ';'.join(reason for mask, reason in zip(masks[1:], ('poor_contact', 'clipping')) if mask[-1])


def score(bundle, frame, metadata, policy, end_ms, fresh_ms):
    duration = 90 if policy == 'baseline' else int(policy)
    minimum = 85 if policy == 'baseline' else duration
    selected = trailing_frame(frame, end_ms, duration, fresh_ms)
    row = dict(available=False, status='insufficient_data', sbp=np.nan, dbp=np.nan,
               accepted_windows=0, clean_coverage_s=0., analyzer_status='not_run',
               analyzer_reason='', reason='', input_start_ms=np.nan, input_end_ms=np.nan)
    if len(selected) < 2:
        return row
    span = (selected.timestamp_ms.iloc[-1] - selected.timestamp_ms.iloc[0]) / 1000
    row.update(input_start_ms=float(selected.timestamp_ms.iloc[0]),
               input_end_ms=float(selected.timestamp_ms.iloc[-1]))
    # One nominal sample tolerance only for trailing slices; no fabricated samples.
    if span + (0.011 if policy != 'baseline' else 0) < minimum:
        return row
    meta = prefix_metadata(metadata, end_ms, row['input_start_ms'])
    diagnostic = analyze_upper_arm_ppg(selected, meta)
    row.update(analyzer_status=diagnostic.status, analyzer_reason=diagnostic.status_reason)
    if policy != 'baseline':
        contact_reason = current_contact_reason(selected, meta, bundle.config)
        if contact_reason:
            row.update(status='contact_artifact', reason='current_sample:' + contact_reason)
            return row
    result = predict_frame(bundle, selected, meta)
    row.update(available=result.numeric_available, status=result.status, reason=result.reason,
               sbp=result.sbp if result.numeric_available else np.nan,
               dbp=result.dbp if result.numeric_available else np.nan,
               accepted_windows=result.accepted_windows, clean_coverage_s=result.clean_coverage_s)
    return row


def error_metrics(predictions, labels):
    errors = np.asarray(predictions, float) - np.asarray(labels, float)
    if not len(errors):
        return dict(count=0, mae=None, rmse=None, bias=None, maximum_absolute_error=None)
    return dict(count=len(errors), mae=float(np.mean(abs(errors))),
                rmse=float(np.sqrt(np.mean(errors ** 2))), bias=float(np.mean(errors)),
                maximum_absolute_error=float(np.max(abs(errors))))


def meets_criteria(candidate, baseline, retention):
    return bool(candidate['count'] > 0 and baseline['count'] == candidate['count']
                and retention is not None and retention >= .8
                and candidate['mae'] <= baseline['mae'] + 2
                and candidate['maximum_absolute_error'] <= baseline['maximum_absolute_error'] + 5)


def evaluate(updates, endpoints, calibration):
    """One row per cuff occasion; matched methods use exactly the same endpoints."""
    if endpoints.duplicated(['occasion_id', 'policy']).any():
        raise ValueError('Duplicate cuff occasion: do not count repeated windows as labels')
    metric_rows, decisions = [], []
    base_u = updates[updates.policy == 'baseline']
    base_e = endpoints[endpoints.policy == 'baseline']
    for duration in DURATIONS:
        policy = str(duration)
        joined = base_u.merge(updates[updates.policy == policy], on=['recording', 'end_ms'], suffixes=('_b', '_c'))
        denominator = int(joined.available_b.sum())
        retained = int((joined.available_b & joined.available_c).sum())
        retention = retained / denominator if denominator else None
        paired = base_e.merge(endpoints[endpoints.policy == policy], on='occasion_id', suffixes=('_b', '_c'))
        paired = paired[paired.available_b & paired.available_c & paired.label_eligible_b & paired.label_eligible_c]
        passed = []
        for target in ('sbp', 'dbp'):
            labels = paired[f'reference_{target}_b']
            b = error_metrics(paired[f'{target}_b'], labels)
            c = error_metrics(paired[f'{target}_c'], labels)
            passed.append(meets_criteria(c, b, retention))
            for method, metrics in [('baseline_matched', b), ('candidate_matched', c),
                                    ('zero_change_matched', error_metrics(np.full(len(labels), calibration[target]), labels))]:
                metric_rows.append(dict(duration_s=duration, target=target, method=method, **metrics))
        # Standalone endpoint metrics expose candidate-only acceptance, without
        # comparing mismatched sets as if they were paired performance.
        for method, source in [('candidate_all', endpoints[endpoints.policy == policy]), ('baseline_all', base_e)]:
            eligible = source[source.available & source.label_eligible]
            for target in ('sbp', 'dbp'):
                metric_rows.append(dict(duration_s=duration, target=target, method=method,
                                        **error_metrics(eligible[target], eligible[f'reference_{target}'])))
        decisions.append(dict(duration_s=duration, matched_cuff_occasions=len(paired),
                              baseline_updates=denominator, retained_updates=retained,
                              retention=retention, candidate_only_updates=int((~joined.available_b & joined.available_c).sum()),
                              passes=all(passed)))
    return pd.DataFrame(metric_rows), pd.DataFrame(decisions)


def replay(frame, metadata, bundle, recording):
    first, last = float(frame.timestamp_ms.iloc[0]), float(frame.timestamp_ms.iloc[-1])
    runs = recording_runs(metadata, first, last)
    bundles = {'baseline': bundle, **{str(d): short_bundle(bundle, d) for d in DURATIONS}}
    updates, endpoints = [], []
    # Endpoint is evaluated separately; it is not added to the four-second grid.
    times = [(float(x), False) for x in np.arange(first, last + .001, 4000)] + [(last, True)]
    for end, endpoint in times:
        run = next(((i, a, b) for i, (a, b) in enumerate(runs) if a <= end < b or a <= end == b == last), None)
        for policy in POLICIES:
            if run is None:
                result = dict(available=False, status='motion_or_unknown', sbp=np.nan, dbp=np.nan,
                              analyzer_status='not_run', reason='moving, stale or missing firmware motion status')
            else:
                result = score(bundles[policy], frame, metadata, policy, end, run[1])
            row = dict(recording=recording, policy=policy, end_ms=end,
                       elapsed_s=(end-first)/1000, run_index=run[0] if run else -1,
                       since_still_s=(end-run[1])/1000 if run else np.nan,
                       # Latency is diagnostic: existing pilots may never satisfy
                       # the production initial 85-second warm-up before movement.
                       initial_warmup_elapsed=bool(run and any(b-a >= 85000 for a,b in runs[:run[0]])),
                       **result)
            (endpoints if endpoint else updates).append(row)
    return updates, endpoints, runs


def write_summary(output, updates, endpoints, decisions, inventory, recovery):
    selected = decisions.loc[decisions.passes, 'duration_s'].tolist()
    lines = ['# Offline recovery-duration comparison', '',
             'Development evidence only. Original model, calibration and preprocessing frozen; no live changes.', '',
             f'Recordings considered: {len(inventory)}; replayed: {sum(x["included"] for x in inventory)}.',
             f'Selected duration: {min(selected) if selected else "none — no candidate meets all criteria"}.', '',
             '## Decisions', '', decisions.to_string(index=False), '',
             'See metrics.csv for matched SBP/DBP errors and the zero-change baseline. '
             'With so few baseline-accepted update times and matched cuff occasions, '
             'a pass/fail is a small development comparison, not a precise performance estimate.', '',
             '## Interpretation', '',
             'Cuff errors use one final pre-cuff endpoint per occasion, restricted to documented after-PPG labels. '
             'Calibration and model-fitting occasions are excluded from the decision metrics. Other previously '
             'examined occasions remain development evidence, not a new untouched test.',
             'Retention uses the identical four-second update grid. Coverage holds an accepted update for at most '
             'four seconds, truncated at movement/staleness/end; unavailable updates never reuse BP. '
             'Short-window acceptance bypasses only the whole-occasion upper-arm gate in an isolated policy; '
             'its full diagnostic result is reported alongside unchanged per-window quality checks.',
             'Recovery latencies start at the later of firmware Still status and the guarded recovery cue interval, not observed action onset. They are diagnostic '
             'opportunities, not proof of a deployed initial-warmup/recovery state machine or BP accuracy. '
             'The five-minute pilot began with less than 85 seconds still, so it cannot validate that initial-warmup sequence.',
             'Hardware/stream faults exclude complete recordings conservatively because metadata counters cannot '
             'reconstruct their exact historical occurrence. No future signal samples or motion messages enter a scored slice.',
             'Known ambiguous visual labels and the shadow classifier are not used as gates. Moderate/severe motion '
             'and BP during movement are outside this experiment.', '',
             'If no duration passes, do not collect the six validation occasions or enable faster live recovery yet. '
             'Inspect the rejection breakdown and matched metrics to distinguish poor coverage from BP error.', '',
             'See decisions.csv, metrics.csv, recording_results.csv, endpoint_predictions.csv, '
             'recovery_latency.csv, rejection_counts.csv, and replay_updates.csv. No models were trained.']
    (output/'conclusion.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default=MODEL)
    parser.add_argument('--output-dir', default='data/processed/bp_recovery_comparison')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(args.model_dir, expected_participant_id='P001')
    paths = list(Path(args.model_dir).glob('*'))
    paths = [p for p in paths if p.is_file()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest_metrics = json.loads((Path(args.model_dir)/'metrics.json').read_text())
    fitting = set(manifest_metrics['split']['development_occasion_ids']) | {bundle.calibration_id}
    records = [r for r in _discover_local(bundle.config['datasets']['local_upper_arm']) if r.participant_id == 'P001']
    sources = [(Path(r.metadata_path), r) for r in records]
    sources.append((Path('data/raw')/(PILOT+'_metadata.json'), None))
    updates, endpoints, inventory, recovery = [], [], [], []
    for index, (path, rec) in enumerate(sources):
        metadata = read_metadata(path)
        csv_path = path.with_name(path.name.replace('_metadata.json', '_ppg.csv'))
        name = csv_path.stem.removesuffix('_ppg')
        print(f'{index+1}/{len(sources)} {name}', flush=True)
        entry = dict(recording=name, included=False, reason='', label_timing=rec.label_timing if rec else '',
                     occasion_id=rec.label_group_id if rec else '', metadata_path=str(path))
        inventory.append(entry)
        if not csv_path.exists():
            entry['reason'] = 'missing_ppg'; continue
        for source in (path, csv_path):
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = pd.read_csv(csv_path)
        if len(frame) < 2:
            entry['reason'] = 'insufficient_data'; continue
        # Do not inherit manual quality labels as universal exclusions. Signal
        # health and continuity are checked independently of BP or visual labels.
        faults = recording_quality_reasons(_recording_context('P001'), signal_from_ppg_frame(frame, metadata))
        if faults:
            entry['reason'] = ';'.join(faults); continue
        if not metadata.get('firmware_motion_updates'):
            entry['reason'] = 'missing_firmware_motion_status'; continue
        # Cuff-labelled motion protocols are not stationary/recovery BP references.
        if rec and any(word in rec.session_id.lower() for word in ('motion', 'movement')):
            entry['reason'] = 'movement_protocol_not_endpoint_labelled'; continue
        entry['included'] = True
        u, e, runs = replay(frame, metadata, bundle, name)
        updates.extend(u)
        for row in e:
            row.update(occasion_id=rec.label_group_id if rec else name,
                       reference_sbp=rec.sbp if rec else np.nan, reference_dbp=rec.dbp if rec else np.nan,
                       label_eligible=bool(rec and rec.label_timing == 'after_ppg' and rec.label_group_id not in fitting),
                       fitting_occasion=bool(rec and rec.label_group_id in fitting),
                       label_timing=rec.label_timing if rec else 'none')
        endpoints.extend(e)
        if rec is None:
            for i, (start, end) in enumerate(runs):
                if i == 0: continue
                for policy in POLICIES:
                    accepted = [x for x in u if x['run_index'] == i and x['policy'] == policy and x['available']]
                    recovery.append(dict(recording=name, run_index=i, policy=policy, still_start_ms=start,
                                         still_duration_s=(end-start)/1000,
                                         resume_latency_s=accepted[0]['since_still_s'] if accepted else np.nan,
                                         resumed=bool(accepted), initial_warmup_preceded=any(b-a>=85000 for a,b in runs[:i])))
        # Flush progress so a long offline run is inspectable/restartable.
        pd.DataFrame(updates).to_csv(output/'replay_updates.csv', index=False)
    u, e = pd.DataFrame(updates), pd.DataFrame(endpoints)
    if u.empty:
        raise RuntimeError('No replayable recordings; inspect raw data and firmware motion status')
    metrics, decisions = evaluate(u, e, {'sbp':bundle.calibration_sbp, 'dbp':bundle.calibration_dbp})
    recording_rows = []
    for (recording, policy), group in u.groupby(['recording', 'policy'], sort=False):
        available = group[group.available]
        # Each output expires at the next tick, first non-still message or stale
        # boundary. This counts estimate availability, not source-window duration.
        meta_path = next(Path(x['metadata_path']) for x in inventory if x['recording']==recording)
        meta = read_metadata(meta_path)
        first = float(group.end_ms.iloc[0]); last = float(e[e.recording==recording].end_ms.iloc[0])
        runs = recording_runs(meta, first, last)
        coverage = sum(max(0, min(x.end_ms+4000, last, next((b for a,b in runs if a<=x.end_ms<=b),x.end_ms))-x.end_ms)
                       /1000 for x in available.itertuples())
        recording_rows.append(dict(recording=recording, policy=policy, updates=len(group), accepted_updates=len(available),
                                   coverage_s=coverage, coverage_fraction=coverage/((last-first)/1000),
                                   sbp_std=float(available.sbp.std(ddof=0)) if len(available) else np.nan,
                                   dbp_std=float(available.dbp.std(ddof=0)) if len(available) else np.nan))
    for name, data in [('recording_inventory',pd.DataFrame(inventory)), ('endpoint_predictions',e),
                       ('metrics',metrics), ('decisions',decisions), ('recovery_latency',pd.DataFrame(recovery)),
                       ('recording_results',pd.DataFrame(recording_rows)),
                       ('rejection_counts',u.groupby(['policy','status']).size().reset_index(name='updates'))]:
        data.to_csv(output/(name+'.csv'), index=False)
    base = u[u.policy=='baseline']
    differences=[]
    for policy in map(str,DURATIONS):
        paired=base.merge(u[u.policy==policy],on=['recording','end_ms'],suffixes=('_baseline','_candidate'))
        paired=paired[paired.available_baseline & paired.available_candidate].copy()
        paired['policy']=policy
        for target in ('sbp','dbp'):
            paired[target+'_difference']=paired[target+'_candidate']-paired[target+'_baseline']
        differences.append(paired[['recording','end_ms','policy','sbp_difference','dbp_difference']])
    pd.concat(differences).to_csv(output/'matched_update_differences.csv',index=False)
    for source in Path('data/labels').glob('*.csv'):
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for source in list(Path('tools/bp_core').glob('*.py')) + [Path('tools/upper_arm_hr.py')]:
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    hashes[str(ACTIVITIES)] = hashlib.sha256(ACTIVITIES.read_bytes()).hexdigest()
    hashes[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for source in paths:
        assert hashes[str(source)] == hashlib.sha256(source.read_bytes()).hexdigest(), 'Frozen model changed'
    (output/'provenance.json').write_text(json.dumps(dict(source_sha256=hashes, research_only=True,
        duration_candidates=list(DURATIONS), step_s=4, policy='3 windows and 80% unique coverage; whole analyzer diagnostic only'),indent=2))
    write_summary(output,u,e,decisions,inventory,recovery)
    print(decisions.to_string(index=False),flush=True)
    print(f'Results: {output.resolve()}',flush=True)


if __name__ == '__main__':
    main()
