import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_bp import (ROOT, ablation_comparison, accepted_unique_seconds, audit,
                       assert_split, attach_motion_band_features, coverage,
                       evaluate_ablation, extract_recording, feature_sets,
                       interval_union, load_config, metrics, movement_duration,
                       quality_decision, stream_reasons, timing_reasons,
                       validate_reference, PPG_COLUMNS, chronological_split,
                       prediction_report)
from test_motion_quality import write_trial


def examples(participant, start, count=10):
    rows = []
    for i in range(start, start + count):
        time = pd.Timestamp('2026-01-01', tz='UTC') + pd.Timedelta(days=i)
        rows.append(dict(participant_id=participant, recording_id=f'{participant}_{i}',
            recording_sha256=f'{participant}_{i}_hash', cuff_occasion_id=f'{participant}_{i}',
            support_start_utc=str(time), support_end_utc=str(time + pd.Timedelta(seconds=20)),
            calibration_end_utc='2025-01-01T00:00:00Z', calibration_occasion_id=participant+'_cal',
            reference_kind='continuous', reference_valid=True, reference_alignment_verified=True,
            reference_source_sha256='a'*64, reference_start_ms=0., reference_end_ms=20000.,
            start_timestamp_ms=0., end_timestamp_ms=8000., true_sbp=120., true_dbp=80.,
            baseline_sbp=120., baseline_dbp=80., morph__rise_time_s=.2,
            calibration__rise_time_s=.2, imu_activity_mean_g=.04,
            imu_dynamic_rms_g=.04, imu_movement_duration_s=4.,
            severity='mild', signal_eligible=True))
    return pd.DataFrame(rows)


class MotionBPTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'config/motion_bp_v1.json')

    def test_timestamp_duration_and_union(self):
        duration = movement_duration(np.array([0, 100, 300, 700]), np.array([1, 1, 0, 0]), 50, 600, .1)
        self.assertAlmostEqual(duration['imu_movement_duration_s'], .25)
        self.assertAlmostEqual(duration['imu_longest_motion_bout_s'], .25)
        self.assertEqual(interval_union([(0, 8), (4, 12), (20, 24)]), 16)

    def test_timing_is_bp_independent_and_unknown_health_fails(self):
        metadata = {key: 0 for key in ('firmware_i2c_error_count', 'firmware_fifo_overflow_count', 'imu_firmware_i2c_error_count', 'imu_firmware_fifo_overflow_count')}
        self.assertEqual(timing_reasons(metadata, self.config), [])
        metadata['sbp'] = 300
        self.assertEqual(timing_reasons(metadata, self.config), [])
        metadata['firmware_warning_events'] = [{'event': 'timestamp_lag', 'lag_us': 50000}]
        self.assertIn('timestamp_lag_uncertain', timing_reasons(metadata, self.config))
        metadata['firmware_warning_events'] = [{'event': 'ppg_clock_observation_rejected', 'phase_error_us': -50000}]
        self.assertIn('ppg_clock_observation_rejected', timing_reasons(metadata, self.config))
        metadata['firmware_warning_events'] = []
        metadata['firmware_clock_rejected_observation_count'] = 1
        self.assertIn('ppg_clock_observation_rejected', timing_reasons(metadata, self.config))
        self.assertTrue(timing_reasons({}, self.config))

    def test_extraction_unequal_rates_and_faults(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = write_trial(Path(directory))
            ppg, imu = pd.read_csv(trial.ppg_path), pd.read_csv(trial.imu_path)
            # Short bounded excerpt, preserving unequal timestamps and bracketing samples.
            ppg, imu = ppg.iloc[:1701], imu.iloc[:2200]
            result = extract_recording(ppg, imu, trial.metadata, self.config)
            self.assertGreater(len(result), 1)
            self.assertIn('imu_dynamic_rms_g', result)
            self.assertIn('morph__rise_time_s', result)
            self.assertTrue((result.start_timestamp_ms >= 10000).all())
            ppg.loc[100, 'timestamp_ms'] = np.nan
            self.assertIn('nonfinite_stream', stream_reasons(ppg, PPG_COLUMNS, self.config))
            rejected = extract_recording(ppg, imu, trial.metadata, self.config)
            self.assertEqual(rejected.iloc[0].status, 'Insufficient data')

    def test_missing_ppg_never_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = write_trial(Path(directory))
            trial.ppg_path.unlink()
            result = audit(Path(directory), self.config)
            self.assertEqual(result.iloc[0].available_modalities, 'imu_only')
            self.assertFalse(result.iloc[0].paired_feature_eligible)
            self.assertFalse(result.iloc[0].identity_resolved)

    def test_quality_states_and_no_numeric_permission(self):
        features = dict(imu_activity_mean_g=.04, imu_dynamic_rms_g=.04,
            ppg_dc_median=80000., ppg_clipping_fraction=0.,
            ppg_template_correlation=.9, ppg_ibi_cv=.1, ppg_valid_beat_count=8, ppg_morphology_accepted=True)
        result = quality_decision(features, [], self.config)
        self.assertTrue(result['signal_eligible'])
        self.assertEqual(result['status'], 'Low confidence')
        features['true_sbp'] = 300
        self.assertEqual(result, quality_decision(features, [], self.config))
        features['imu_activity_mean_g'] = .3
        self.assertEqual(quality_decision(features, [], self.config)['status'], 'Motion too severe')
        features['ppg_dc_median'] = 1000
        self.assertEqual(quality_decision(features, [], self.config)['status'], 'Poor contact')
        features['imu_activity_mean_g'], features['ppg_dc_median'] = .04, 80000
        features['ppg_template_correlation'] = None
        self.assertFalse(quality_decision(features, [], self.config)['signal_eligible'])

    def test_reference_refuses_cuff_propagation_and_partial_coverage(self):
        frame = examples('P1', 1)
        validate_reference(frame)
        frame.loc[0, 'reference_kind'] = 'cuff'
        with self.assertRaises(ValueError): validate_reference(frame)
        frame.loc[0, 'reference_kind'] = 'continuous'
        frame.loc[0, 'reference_end_ms'] = 4000
        with self.assertRaises(ValueError): validate_reference(frame)

    def test_recording_cuff_hash_participant_and_time_leakage(self):
        train, test = examples('P1', 1), examples('P1', 20)
        assert_split(train, test)
        for column in ['recording_id', 'recording_sha256', 'cuff_occasion_id']:
            invalid = test.copy()
            invalid.loc[0, column] = train.loc[0, column]
            with self.assertRaises(ValueError): assert_split(train, invalid)
        with self.assertRaises(ValueError): assert_split(train, test, held_out_participant=True)
        with self.assertRaises(ValueError): assert_split(test, train)

    def test_metrics_and_coverage_count_unique_time(self):
        result = metrics([100, 110], [102, 106])
        self.assertEqual(result['mae'], 3.)
        self.assertAlmostEqual(result['rmse'], np.sqrt(10))
        self.assertEqual(result['bias'], -1.)
        self.assertEqual(result['maximum_absolute_error'], 4.)
        frame = pd.DataFrame(dict(recording_id=['a', 'a'], start_timestamp_ms=[0, 4000], end_timestamp_ms=[8000, 12000]))
        self.assertAlmostEqual(coverage(frame, [True, False]), 2/3)
        self.assertEqual(metrics([], [])['count'], 0)

    def test_ablation_independent_uncertainty_and_test(self):
        train, uncertainty, test = examples('P1', 1), examples('P1', 20), examples('P1', 40)
        predictions, report = evaluate_ablation(train, uncertainty, test, self.config)
        self.assertEqual(set(predictions.model), {'ppg_only', 'ppg_intensity', 'ppg_full_imu', 'zero_change', 'ppg_imu_rejection_only'})
        self.assertFalse(report['deployment_eligible'])
        self.assertEqual(report['models']['ppg_full_imu']['mild']['sbp']['mae'], 0)
        modified = test.copy()
        modified['true_sbp'] += 10
        new, _ = evaluate_ablation(train, uncertainty, modified, self.config)
        np.testing.assert_array_equal(predictions.predicted_sbp, new.predicted_sbp)
        modified['calibration_end_utc'] = '2030-01-01'
        with self.assertRaises(ValueError): evaluate_ablation(train, uncertainty, modified, self.config)

    def test_feature_selection_excludes_labels_and_identity(self):
        frame = attach_motion_band_features(examples('P1', 1), self.config)
        frame['imu_sbp_label'] = 123
        frame['activity_label'] = 1
        sets = feature_sets(frame)
        for columns in sets.values():
            self.assertNotIn('imu_sbp_label', columns)
            self.assertNotIn('activity_label', columns)
        self.assertFalse(any(column.startswith('imu_') for column in sets['ppg_only']))
        self.assertIn('imu_activity_mean_g', sets['ppg_intensity'])
        self.assertIn('imu_band_mild', sets['ppg_intensity'])
        self.assertIn('imu_dynamic_rms_g', sets['ppg_full_imu'])

    def test_motion_bands_use_shared_stage1_source_and_boundaries(self):
        frame = examples('P1', 1, 4)
        frame['imu_activity_mean_g'] = [.01, .02, .08, .20]
        frame['imu_dynamic_rms_g'] = [.5, .5, .5, .001]
        result = attach_motion_band_features(frame, self.config)
        self.assertEqual(result.severity.tolist(), ['stationary', 'mild', 'moderate', 'severe'])
        self.assertEqual(result.imu_band_mild.tolist(), [0., 1., 0., 0.])
        self.assertEqual(result.imu_band_moderate.tolist(), [0., 0., 1., 0.])
        self.assertEqual(result.imu_band_severe.tolist(), [0., 0., 0., 1.])

    def test_chronological_split_keeps_overlapping_windows_together(self):
        frame = examples('P1', 1, 30)
        duplicate = frame.copy()
        duplicate['start_timestamp_ms'] = 4000.
        duplicate['end_timestamp_ms'] = 12000.
        result = chronological_split(pd.concat([frame, duplicate], ignore_index=True))
        self.assertEqual([len(x) for x in result.values()], [20, 20, 20])
        with self.assertRaises(ValueError): chronological_split(frame.iloc[:20])

    def test_all_rejected_and_false_acceptance_reporting(self):
        frame = examples('P1', 1)
        frame['accepted'] = False
        frame['predicted_sbp'], frame['predicted_dbp'] = 120., 80.
        frame['half_width_sbp'], frame['half_width_dbp'] = 1., 1.
        frame['reviewed_quality'] = 'severely_corrupted'
        report = prediction_report(frame, 10.)
        self.assertEqual(report['all']['coverage'], 0.)
        self.assertIsNone(report['all']['sbp']['mae'])
        self.assertEqual(report['false_acceptance_severely_corrupted']['rate'], 0.)
        frame.loc[0, 'accepted'] = True
        self.assertEqual(prediction_report(frame, 10.)['false_acceptance_severely_corrupted']['rate'], .1)

    def test_held_out_participant_and_inconsistent_calibration(self):
        train, uncertainty, test = examples('P1', 1), examples('P1', 20), examples('P2', 40)
        predictions, _ = evaluate_ablation(train, uncertainty, test, self.config, held_out_participant=True)
        self.assertGreater(len(predictions), 0)
        test.loc[1, 'baseline_sbp'] += 1
        with self.assertRaises(ValueError): evaluate_ablation(train, uncertainty, test, self.config, held_out_participant=True)

    def test_unseen_motion_severity_cannot_get_narrow_interval(self):
        train, uncertainty, test = examples('P1', 1), examples('P1', 20), examples('P1', 40)
        test['imu_activity_mean_g'] = .1
        result, report = evaluate_ablation(train, uncertainty, test, self.config)
        self.assertFalse(result.accepted.any())
        self.assertIsNone(report['models']['ppg_full_imu']['moderate']['sbp']['mae'])

    def test_ablation_comparison_uses_common_windows_and_unique_coverage(self):
        base = examples('P1', 1, 3)
        base['recording_id'] = 'P1_shared_recording'
        base['start_timestamp_ms'] = [0., 4000., 12000.]
        base['end_timestamp_ms'] = [8000., 12000., 20000.]
        base['model'] = 'ppg_only'
        base['accepted'] = [True, True, False]
        base['predicted_sbp'] = [124., 124., 124.]
        base['predicted_dbp'] = [84., 84., 84.]
        intensity = base.copy()
        intensity['model'] = 'ppg_intensity'
        intensity['accepted'] = [True, True, True]
        intensity['predicted_sbp'] = [122., 122., 122.]
        intensity['predicted_dbp'] = [82., 82., 82.]
        full = intensity.copy()
        full['model'] = 'ppg_full_imu'
        report = ablation_comparison([base, intensity, full])['ppg_intensity_vs_ppg_only']['mild']
        self.assertEqual(report['common_accepted_window_count'], 2)
        self.assertEqual(report['common_accepted_unique_seconds'], 12.)
        self.assertAlmostEqual(report['ppg_only_accepted_coverage'], .6)
        self.assertEqual(report['candidate_accepted_coverage'], 1.)
        self.assertEqual(report['targets']['sbp']['mae_delta_candidate_minus_ppg_only'], -2.)
        self.assertTrue(report['improves_both_targets_on_common_windows'])
        self.assertEqual(accepted_unique_seconds(base, [False, False, False]), 0.)

    def test_ablation_comparison_reports_unavailable_without_common_windows(self):
        outputs = []
        for name, accepted in [('ppg_only', False), ('ppg_intensity', True), ('ppg_full_imu', True)]:
            frame = examples('P1', 1, 1)
            frame['model'], frame['accepted'] = name, accepted
            frame['predicted_sbp'], frame['predicted_dbp'] = 120., 80.
            outputs.append(frame)
        result = ablation_comparison(outputs)['ppg_intensity_vs_ppg_only']['all']
        self.assertEqual(result['common_accepted_window_count'], 0)
        self.assertIsNone(result['targets']['sbp']['mae_delta_candidate_minus_ppg_only'])
        self.assertFalse(result['improves_both_targets_on_common_windows'])


if __name__ == '__main__':
    unittest.main()
