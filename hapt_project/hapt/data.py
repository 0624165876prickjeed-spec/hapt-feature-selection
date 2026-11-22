import re
import urllib.request
import zipfile

import numpy as np

from .common import digest_file, digest_value, write_csv, write_json


class DataStore:
    def __init__(self, configuration, workspace, scratch, logger):
        self.cfg = configuration
        self.workspace = workspace
        self.scratch = scratch
        self.logger = logger
        self.raw_path = scratch / 'raw'
        self.records = []

    def prepare(self, artifact_directory):
        archive_path = self.workspace / self.cfg['dataset']['local_archive']
        if not archive_path.exists():
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info('Downloading official public archive: %s', archive_path)
            urllib.request.urlretrieve(self.cfg['dataset']['download_url'], archive_path)
        if digest_file(archive_path) != self.cfg['dataset']['sha256']:
            raise ValueError('Archive SHA256 mismatch')
        development = set(self.cfg['split']['development_subjects'])
        test = set(self.cfg['split']['test_subjects'])
        folds = list(map(set, self.cfg['split']['validation_subject_folds']))
        assert not development & test
        assert set.union(*folds) == development
        assert sum(map(len, folds)) == len(development)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.records = []
        window_rows = []
        discarded = {'grid_windows': 0, 'retained': 0}
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError('Archive CRC failure')
            labels = np.loadtxt(archive.open(self.cfg['dataset']['labels_file']), dtype=np.int64)
            allowed = set(map(int, self.cfg['dataset']['classes']))
            for name in sorted(archive.namelist()):
                match = re.fullmatch(r'RawData/acc_exp(\d+)_user(\d+)\.txt', name)
                if match is None:
                    continue
                experiment, subject = map(int, match.groups())
                if subject not in development | test:
                    raise ValueError('Unknown subject')
                partition = 'development' if subject in development else 'test'
                record_id = f'exp{experiment:02d}_user{subject:02d}'
                raw_file = self.raw_path / f'{record_id}.npz'
                gyroscope_name = self.cfg['dataset']['record_pattern_gyro'].format(experiment_id=experiment, subject_id=subject)
                acceleration = np.loadtxt(archive.open(name), dtype=np.float64)
                gyroscope = np.loadtxt(archive.open(gyroscope_name), dtype=np.float64)
                assert acceleration.shape == gyroscope.shape and acceleration.shape[1] == 3
                signals = np.column_stack([acceleration, gyroscope])
                assert np.isfinite(signals).all()
                segments = labels[(labels[:, 0] == experiment) & (labels[:, 1] == subject)]
                ordered = segments[np.argsort(segments[:, 3])]
                assert (ordered[:, 3] >= 0).all() and (ordered[:, 4] <= len(signals)).all()
                assert (ordered[:, 4] > ordered[:, 3]).all()
                assert (ordered[1:, 3] > ordered[:-1, 4]).all()
                starts, targets, internal_mask, total_grid = self.make_windows(len(signals), ordered, allowed)
                if not raw_file.exists():
                    np.savez(raw_file, signals=signals, starts=starts, targets=targets, internal_mask=internal_mask)
                else:
                    with np.load(raw_file) as existing:
                        assert np.array_equal(existing['starts'], starts)
                        assert np.array_equal(existing['targets'], targets)
                        assert np.array_equal(existing['signals'], signals)
                record = {'record_id': record_id, 'experiment_id': experiment, 'subject_id': subject, 'partition': partition, 'samples': len(signals), 'windows': len(starts), 'raw_sha256': digest_file(raw_file)}
                self.records.append(record)
                discarded['grid_windows'] += total_grid
                discarded['retained'] += len(starts)
                for start, target in zip(starts, targets):
                    window_rows.append({'window_id': f'{record_id}_s{start:05d}', 'record_id': record_id, 'experiment_id': experiment, 'subject_id': subject, 'start_sample': int(start), 'end_sample_exclusive': int(start + self.cfg['window']['length_samples']), 'activity': int(target), 'partition': partition})
            (artifact_directory / 'data').mkdir(parents=True, exist_ok=True)
            (artifact_directory / 'data' / 'SOURCE_README.txt').write_bytes(archive.read('README.txt'))
        for partition, expected_key, count_key in [('development', 'expected_development_windows', 'expected_development_class_counts'), ('test', 'expected_test_windows', 'expected_test_class_counts')]:
            selected = [row for row in window_rows if row['partition'] == partition]
            assert len(selected) == self.cfg['window'][expected_key]
            for activity, expected in self.cfg['window'][count_key].items():
                assert sum(row['activity'] == int(activity) for row in selected) == expected
        for fold in folds:
            present = {row['activity'] for row in window_rows if row['subject_id'] in fold}
            assert present == allowed
        write_csv(artifact_directory / 'data' / 'record_manifest.csv', self.records)
        write_csv(artifact_directory / 'data' / 'window_manifest.csv', window_rows)
        write_json(artifact_directory / 'data' / 'split_manifest.json', self.cfg['split'])
        write_json(artifact_directory / 'data' / 'data_audit.json', {'archive_sha256': self.cfg['dataset']['sha256'], 'records': len(self.records), 'window_counts': discarded, 'discarded_grid_windows': discarded['grid_windows'] - discarded['retained'], 'subject_overlap_count': len(development & test), 'record_overlap_count': 0, 'window_rule_hash': digest_value(self.cfg['window']), 'test_access': 'structural_integrity_and_manifest_only_before_freeze'})
        self.logger.info('Data prepared: %s records, %s windows; subject/record overlap=0', len(self.records), len(window_rows))

    def make_windows(self, sample_count, segments, allowed):
        window = self.cfg['window']
        grid = np.arange(window['record_grid_origin'], sample_count - window['length_samples'] + 1, window['stride_samples'], dtype=np.int64)
        targets = np.zeros(len(grid), dtype=np.int64)
        internal_mask = np.zeros(sample_count, dtype=bool)
        for _, _, activity, first, last in segments:
            if int(activity) not in allowed:
                continue
            lower = int(first) + window['guard_samples']
            upper = int(last) - window['guard_samples']
            if upper <= lower:
                continue
            internal_mask[lower:upper] = True
            accepted = (grid >= lower) & (grid + window['length_samples'] <= upper)
            assert not (targets[accepted] > 0).any()
            targets[accepted] = activity
        return grid[targets > 0], targets[targets > 0], internal_mask, len(grid)

    def subset_records(self, subjects, limit_per_subject=None):
        subjects = set(subjects)
        records = [record for record in self.records if record['subject_id'] in subjects]
        if limit_per_subject is None:
            return records
        counts = {}
        selected = []
        for record in records:
            subject = record['subject_id']
            if counts.get(subject, 0) < limit_per_subject:
                selected.append(record)
                counts[subject] = counts.get(subject, 0) + 1
        return selected

    def load(self, record):
        with np.load(self.raw_path / (record['record_id'] + '.npz')) as archive:
            return {name: archive[name] for name in archive.files}

    def reference_power(self, records):
        total = np.zeros(len(self.cfg['dataset']['channels']), dtype=np.float64)
        count = 0
        for record in records:
            loaded = self.load(record)
            selected = loaded['signals'][loaded['internal_mask']]
            if len(selected) == 0:
                continue
            centered = selected - selected.mean(axis=0)
            total += np.sum(centered ** 2, axis=0)
            count += len(selected)
        if count == 0:
            raise ValueError('No training samples for reference power')
        power = total / count
        if not np.isfinite(power).all() or (power <= 0).any():
            raise ValueError('Invalid training reference power')
        return power
