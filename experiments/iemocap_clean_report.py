#!/usr/bin/env python3
"""Strict, append-only audit/report of the F0-initialized IEMOCAP experiment.

Never read historical eval/task_only or eval/c6, silently intersect samples,
substitute surrogate predictions, or select folds based on their performance.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from iemocap_missing_rate import build_schedule
from iemocap_qwen_report import cm_metrics

STATES = ('A', 'V', 'T', 'AV', 'AT', 'VT', 'AVT')
LABELS = ('angry', 'happy', 'neutral', 'sad')
ARMS = ('f0', 'task_only', 'c6')
RATES = tuple(i / 10 for i in range(8))
SEEDS = tuple(range(20260905, 20260910))
METRICS = ('WAR', 'UAR', 'MF1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_predictions(paths, expected):
    """Deduplicate identical successful records; conflicting duplicates abort."""
    seen, failures, duplicates = {}, [], 0
    provenance = []
    for path in paths:
        provenance.append({'path': str(path), 'sha256': sha(path)})
        for record in rows(path):
            sid = record['sample_id']
            if record.get('status') != 'success':
                failures.append({'sample_id': sid, 'source': str(path),
                                 'error': record.get('error')})
                continue
            if sid not in expected or record['gt'] != expected[sid]['gt']:
                raise ValueError(f'Unexpected sample or label: {path}: {sid}')
            for state, value in record['states'].items():
                if state not in STATES or value.get('official_pred') not in LABELS:
                    raise ValueError(f'Invalid official prediction: {sid}/{state}')
                key = (sid, state)
                if key in seen:
                    if seen[key] != value:
                        raise ValueError(f'Conflicting duplicate: {path}: {key}')
                    duplicates += 1
                seen[key] = value
    wanted = {(sid, state) for sid in expected for state in STATES}
    missing = sorted(wanted - set(seen))
    return seen, {'sources': provenance, 'unique_keys': len(seen),
                  'expected_keys': len(wanted), 'identical_duplicate_events': duplicates,
                  'missing_keys': missing, 'historical_error_rows': failures,
                  'complete': not missing}


def audit_adapter(directory, source):
    from safetensors.torch import load_file
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    initial_path = directory / 'initial_adapter/adapter_model.safetensors'
    reference_path = source / 'adapter/adapter_model.safetensors'
    initial_equal = False
    if initial_path.exists():
        a, b = load_file(str(initial_path)), load_file(str(reference_path))
        initial_equal = set(a) == set(b) and all(a[k].equal(b[k]) for k in a)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    result = {'manifest': manifest, 'manifest_sha256': sha(manifest_path) if manifest else None,
              'initial_tensors_exact_F0': initial_equal,
              'F0_parameter_sha256': source_manifest['adapter_sha256'],
              'initial_file_sha256': sha(initial_path) if initial_path.exists() else None,
              'F0_file_sha256': sha(reference_path),
              'complete': bool(manifest and manifest['status'] == 'COMPLETE')}
    if manifest:
        if not initial_equal or manifest['initial_adapter_sha256'] != source_manifest['adapter_sha256']:
            raise ValueError(f'F0 initialization mismatch: {directory}')
        final_path = directory / 'adapter/adapter_model.safetensors'
        final = load_file(str(final_path))
        digest = hashlib.sha256()
        for key, value in sorted(final.items()):
            # PEFT removes the adapter name when serializing its tensor keys.
            name = key.replace('.lora_A.weight', '.lora_A.default.weight').replace(
                '.lora_B.weight', '.lora_B.default.weight')
            digest.update(name.encode())
            digest.update(value.float().numpy().tobytes())
        result['final_parameter_sha256_verified'] = digest.hexdigest()
        result['final_file_sha256'] = sha(final_path)
        if digest.hexdigest() != manifest['adapter_sha256']:
            raise ValueError(f'Final adapter digest mismatch: {directory}')
    return result


def make_counts(expected, predictions):
    """One matrix per real dialogue and condition; masks fixed before bootstrap."""
    grouped = collections.defaultdict(lambda: np.zeros((15, 3, 4, 4), dtype=np.int32))
    legacy = collections.defaultdict(lambda: np.zeros((15, 3, 4, 4), dtype=np.int32))
    schedules = []
    keys = sorted(expected)

    def add(sid, state, condition):
        label = LABELS.index(expected[sid]['gt'])
        dialogue = sid.rsplit('_', 1)[0]
        old_dialogue = '_'.join(sid.split('_')[:2])
        for i, arm in enumerate(ARMS):
            pred = LABELS.index(predictions[arm][sid, state]['official_pred'])
            grouped[dialogue][condition, i, label, pred] += 1
            legacy[old_dialogue][condition, i, label, pred] += 1

    for condition, state in enumerate(STATES):
        for sid in keys:
            add(sid, state, condition)
    for ri, rate in enumerate(RATES):
        for seed in SEEDS:
            sd = int(hashlib.sha256(f'iemocap|{rate:.4f}|{seed}'.encode()).hexdigest()[:16], 16)
            assignment, dropped, total = build_schedule(keys, rate, sd)
            schedules.append({'target_rate': rate, 'seed': seed, 'achieved_rate': dropped / total,
                              'assignment': assignment,
                              'sha256': hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()})
            for sid, state in assignment.items():
                add(sid, state, 7 + ri)
    return grouped, legacy, schedules


def deltas(cm):
    values = cm_metrics(cm)
    return np.stack((values[..., 2, :] - values[..., 1, :],
                     values[..., 2, :] - values[..., 0, :]), axis=-2)


def summarize(grouped, n_boot, seed=20260906):
    matrices = np.stack([grouped[key] for key in sorted(grouped)])
    total = matrices.sum(0)
    point, values = deltas(total), cm_metrics(total)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, 15, 2, 3))
    for start in range(0, n_boot, 100):
        n = min(100, n_boot - start)
        weights = rng.multinomial(len(matrices), np.ones(len(matrices)) / len(matrices), size=n)
        samples[start:start+n] = deltas(np.einsum('bg,gcaij->bcaij', weights, matrices))
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    mean_point = point[7:].mean(0)
    mean_lo, mean_hi = np.percentile(samples[:, 7:].mean(1), [2.5, 97.5], axis=0)

    def comparison_record(p, lower, upper):
        return {name: {metric: {'point': float(p[j, k]),
                               'ci95': [float(lower[j, k]), float(upper[j, k])],
                               'excludes_zero': bool(lower[j, k] > 0 or upper[j, k] < 0)}
                       for k, metric in enumerate(METRICS)}
                for j, name in enumerate(('c6_minus_task_only', 'c6_minus_f0'))}

    conditions = {}
    for i, condition in enumerate(list(STATES) + [f'r{round(r*10000):05d}' for r in RATES]):
        conditions[condition] = {
            'metrics': {arm: dict(zip(METRICS, values[i, j].tolist())) for j, arm in enumerate(ARMS)},
            'paired': comparison_record(point[i], lo[i], hi[i])}
    return {'n_dialogues': len(matrices), 'n_samples': int(total[0, 0].sum()),
            'state_wise': {key: conditions[key] for key in STATES},
            'random_missing_rate': {key: value for key, value in conditions.items() if key.startswith('r')},
            'rate_mean_metrics': {arm: dict(zip(METRICS, values[7:, j].mean(0).tolist()))
                                  for j, arm in enumerate(ARMS)},
            'rate_mean_paired': comparison_record(mean_point, mean_lo, mean_hi),
            'all_three_positive_vs_both_controls': bool((mean_point > 0).all())}


def add_raw_reference(result, grouped):
    values = cm_metrics(sum(grouped.values()))[:, 0]
    for i, state in enumerate(STATES):
        result['state_wise'][state]['metrics']['raw_qwen'] = dict(zip(METRICS, values[i].tolist()))
    for i, rate in enumerate(RATES):
        result['random_missing_rate'][f'r{round(rate*10000):05d}']['metrics']['raw_qwen'] = dict(
            zip(METRICS, values[7+i].tolist()))
    result['rate_mean_metrics']['raw_qwen'] = dict(zip(METRICS, values[7:].mean(0).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('work_dir/iemocap_qwen_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-boot', type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Output directory already exists; choose a new append-only version')
    report = {'schema': 'iemocap_clean_audit_v1', 'folds': {}, 'n_boot': args.n_boot,
              'primary_estimand': 'per rate: confusion counts pooled over five fixed mask seeds; then mean over eight rates',
              'bootstrap': 'paired real-dialogue clusters; same resample across all rates; mask seeds are not independent samples',
              'legacy_bootstrap': 'coarser first-two-underscore-fields groups retained as sensitivity only',
              'official_MELD_dev_test_access_this_run': 0}
    protocol_path = args.root / 'protocol/protocol.json'
    report['protocol_sha256'] = sha(protocol_path)
    if report['protocol_sha256'] != '73453b06ab9d6fe736cad5f7348492df8a1b9c42d3cdf7be275c8258e78803f8':
        raise ValueError('Frozen protocol SHA mismatch')
    pooled = collections.defaultdict(lambda: np.zeros((15, 3, 4, 4), dtype=np.int32))
    pooled_raw = collections.defaultdict(lambda: np.zeros((15, 3, 4, 4), dtype=np.int32))
    complete = []
    schedule_output = {}
    for fold in (1, 2, 3):
        directory = args.root / f'fold{fold}'
        manifest_path = args.root / 'protocol' / f'fold{fold}_tst_manifest.jsonl'
        source_rows = rows(manifest_path)
        expected = {row['sample_id']: row for row in source_rows}
        train_path = args.root / 'protocol' / f'fold{fold}_trn_manifest.jsonl'
        train_rows = rows(train_path)
        if len(expected) != len(source_rows) or len({r['sample_id'] for r in train_rows}) != len(train_rows):
            raise ValueError('Duplicate sample in data manifest')
        if len(train_rows) != (4508 if fold == 3 else 4446):
            raise ValueError('Unexpected training fold size')
        if set(expected) & {row['sample_id'] for row in train_rows}:
            raise ValueError('Train/test sample leakage')
        entry = {'n_train': len(train_rows), 'n_test': len(expected),
                 'train_manifest_sha256': sha(train_path), 'test_manifest_sha256': sha(manifest_path),
                 'class_counts': dict(collections.Counter(row['gt'] for row in source_rows)),
                 'initialization': {}, 'evaluation_integrity': {}}
        for arm in ('task_only', 'c6'):
            entry['initialization'][arm] = audit_adapter(directory / 'clean' / arm, directory / 'f0')
        predictions = {}
        for arm in ARMS:
            if arm == 'f0':
                # Use exactly the known two original shards, plus explicitly named repair output.
                paths = [directory / 'eval' / f'f0_shard{s}.jsonl' for s in (0, 1)]
                paths += [directory / 'eval_audited_v1/f0_missing.jsonl']
            else:
                paths = [directory / 'eval_clean' / f'{arm}.jsonl']
                paths += [directory / 'eval_audited_v1' / f'{arm}.jsonl']
            paths = [path for path in paths if path.exists()]
            predictions[arm], entry['evaluation_integrity'][arm] = load_predictions(paths, expected)
        raw_path = directory / 'eval/raw_qwen.jsonl'
        raw, entry['evaluation_integrity']['raw_qwen'] = load_predictions(
            [raw_path] if raw_path.exists() else [], expected)
        available = all(x['complete'] for x in entry['evaluation_integrity'].values()) and all(
            x['complete'] for x in entry['initialization'].values())
        if available:
            left, right = [entry['initialization'][a]['manifest'] for a in ('task_only', 'c6')]
            for field in ('initial_adapter_sha256', 'seed', 'epochs', 'torch_seed', 'incomplete_states', 'pos_weight', 'counters'):
                if left[field] != right[field]:
                    raise ValueError(f'Matched pair mismatch in fold{fold}: {field}')
            grouped, legacy, schedules = make_counts(expected, predictions)
            entry['result'] = summarize(grouped, args.n_boot)
            raw_grouped, _, _ = make_counts(expected, {arm: raw for arm in ARMS})
            add_raw_reference(entry['result'], raw_grouped)
            entry['legacy_grouping_sensitivity'] = summarize(legacy, args.n_boot)
            schedule_output[str(fold)] = schedules
            for key, value in grouped.items():
                pooled[key] += value  # Same dialogue in folds 1 and 2 remains one cluster.
            for key, value in raw_grouped.items():
                pooled_raw[key] += value
            complete.append(fold)
            print(f"fold{fold}: {entry['result']['rate_mean_paired']['c6_minus_task_only']}", flush=True)
        entry['status'] = 'COMPLETE' if available else 'INCOMPLETE'
        report['folds'][str(fold)] = entry
    report['complete_folds'] = complete
    report['completion_status'] = 'COMPLETE' if complete == [1, 2, 3] else 'PENDING'
    report['frozen_criterion_status'] = ('NOT MET' if any(
        not report['folds'][str(f)]['result']['all_three_positive_vs_both_controls'] for f in complete)
        else ('PASS' if complete == [1, 2, 3] else 'PENDING'))
    if complete == [1, 2, 3]:
        report['pooled'] = summarize(pooled, args.n_boot)
        add_raw_reference(report['pooled'], pooled_raw)
    report['next_action'] = ('APPEND_FINAL_SSOT_THEN_BOUNDED_LAMBDA_PILOT' if complete == [1, 2, 3]
                             else 'COMPLETE_EXISTING_CLEAN_EXPERIMENT_AND_MISSING_EVALUATION')
    write_new(args.output_dir / 'report.json', report)
    write_new(args.output_dir / 'schedules.json', schedule_output)
    print(json.dumps({k: report[k] for k in ('completion_status', 'frozen_criterion_status', 'next_action')}))


if __name__ == '__main__':
    main()
