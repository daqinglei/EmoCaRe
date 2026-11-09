"""Opt-in run provenance and real step-0 gate, preserving training RNG state."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from iemocap_clean_report import sha, write_new


def preflight(args, model, processor, runtime, initial, label_names):
    from run_meld_qwen_lora_sft import decode_audio, decode_video
    from run_meld_qwen_sft_selection_eval import prepare_view
    from run_planb_meld_qwen_pilot import cached_sequence_scores

    if not args.init_adapter or not args.require_init_sha or not args.init_gate_reference:
        raise ValueError('Audited preflight requires F0 adapter, required SHA, and reference scores')
    reference_path = Path(args.init_gate_reference)
    reference_meta = json.loads(Path(str(reference_path) + '.started.json').read_text())
    if reference_meta['protocol_sha256'] != sha(args.protocol) or reference_meta['manifest_sha256'] != sha(args.manifest):
        raise ValueError('Step-0 reference was scored under a different protocol or training manifest')
    reference_adapter = Path(reference_meta['args']['adapter']).resolve()
    if sha(reference_adapter / 'adapter_model.safetensors') != sha(Path(args.init_adapter) / 'adapter_model.safetensors'):
        raise ValueError('Step-0 reference adapter differs from the initialization adapter')
    all_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = all_rows[:12]
    references = [json.loads(line) for line in reference_path.read_text().splitlines() if line.strip()]
    if len(references) != 12 or any(x.get('status') != 'success' for x in references):
        raise ValueError('Step-0 reference must have 12 successful unique training samples')
    reference = {r['sample_id']: r for r in references}
    if set(reference) != {r['sample_id'] for r in selected}:
        raise ValueError('Step-0 reference does not use the fixed first 12 training samples')
    anchor_audit = {}
    for path, field in ((args.frozen_margins, 'margin'), (args.frozen_logits, 'class_logits')):
        cache = {}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record['sample_id'], record['state'])
            if key in cache and cache[key] != record[field]:
                raise ValueError(f'Conflicting frozen anchor: {path}: {key}')
            cache[key] = record[field]
        needed = {(r['sample_id'], s) for r in all_rows for s in args.incomplete_states.split(',')}
        if needed - set(cache):
            raise ValueError(f'Incomplete frozen anchor: {path}: {len(needed-set(cache))} missing keys')
        for row in selected:
            for state in args.incomplete_states.split(','):
                if not np.allclose(cache[row['sample_id'], state],
                                   reference[row['sample_id']]['states'][state][field], atol=1e-3, rtol=0):
                    raise ValueError(f'Frozen anchor differs from F0 reference: {row["sample_id"]}/{state}/{field}')
        anchor_audit[field] = {'required_unique_keys': len(needed), 'available_unique_keys': len(cache),
                               'first_12_samples_match_F0_reference': True}
    order = np.random.default_rng(args.seed).permutation(len(all_rows)).tolist() * args.epochs
    if args.max_units:
        order = order[:args.max_units]
    order = order[:len(order) // args.grad_accum * args.grad_accum] or order
    files = [args.protocol, args.manifest, Path(args.init_adapter) / 'adapter_model.safetensors',
             Path(args.init_adapter) / 'adapter_config.json', reference_path,
             Path(str(reference_path) + '.started.json'),
             Path(__file__), Path(__file__).with_name('run_meld_c6_balanced_evidence.py'),
             Path(__file__).with_name('run_meld_qwen_lora_sft.py'),
             Path(__file__).with_name('eval_c6_states.py')]
    files += [p for p in (args.frozen_margins, args.frozen_logits) if p]
    source_protocol_path = Path(runtime['source']['platform_protocol'])
    files.append(source_protocol_path)
    provenance = {'status': 'FROZEN_BEFORE_OPTIMIZER', 'created_unix': time.time(),
                  'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'initial_adapter_sha256': initial, 'file_sha256': {str(p): sha(p) for p in files},
                  'anchor_integrity': anchor_audit,
                  'runtime_training': runtime['training'],
                  'actual_optimizer': {'name': 'AdamW', 'lr': args.lr, 'betas': [0.9, 0.999],
                                       'eps': 1e-8, 'weight_decay': 0.0, 'clip_grad_norm': 1.0,
                                       'scheduler': 'linear decay, no warmup'},
                  'n_train_samples': len(all_rows), 'n_unique_exposed_samples': len(set(order)),
                  'units': len(order), 'optimizer_steps': (len(order)+args.grad_accum-1)//args.grad_accum,
                  'unit_order_sha256': hashlib.sha256(json.dumps(
                      [all_rows[i]['sample_id'] for i in order], separators=(',', ':')).encode()).hexdigest(),
                  'state_order': args.incomplete_states.split(',') + ['AVT'],
                  'official_MELD_dev_test_access': 0}
    write_new(args.output_dir / 'run_provenance.json', provenance)
    write_new(args.output_dir / 'config.json', provenance['args'])
    write_new(args.output_dir / 'manifest.pretrain.json', {
        'status': 'INITIALIZED_GATE_PENDING', 'initial_adapter_sha256': initial,
        'run_provenance_sha256': sha(args.output_dir / 'run_provenance.json')})

    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    numpy_state, python_state = np.random.get_state(), random.getstate()
    training = model.training
    model.eval()
    records = []
    source = json.loads(source_protocol_path.read_text())
    vp = int(runtime['training']['video_total_pixels'])
    try:
        with torch.no_grad():
            for row in selected:
                waveform = decode_audio(Path(row['video_path']))
                video = decode_video(Path(row['video_path']), vp)
                for state in provenance['state_order']:
                    prepared, metadata = prepare_view(source, processor, dict(row), state, waveform, video)
                    scores, _ = cached_sequence_scores(model, prepared, metadata['candidate_token_ids'])
                    target = reference[row['sample_id']]['states'][state]['candidate_scores']
                    a = np.array([scores[name] for name in label_names], dtype=np.float64)
                    b = np.array([target[name] for name in label_names], dtype=np.float64)
                    logp = a - np.logaddexp.reduce(a)
                    logq = b - np.logaddexp.reduce(b)
                    records.append({'sample_id': row['sample_id'], 'state': state,
                                    'max_abs_dlogscore': float(np.max(np.abs(a-b))),
                                    'max_abs_dlogp': float(np.max(np.abs(logp-logq))),
                                    'kl': float(np.sum(np.exp(logp) * (logp-logq))),
                                    'argmax_equal': bool(a.argmax() == b.argmax())})
    finally:
        model.train(training)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    gate = {'n_pairs': len(records), 'max_abs_dlogscore': max(r['max_abs_dlogscore'] for r in records),
            'max_abs_dlogp': max(r['max_abs_dlogp'] for r in records),
            'max_kl': max(r['kl'] for r in records),
            'argmax_agreement': float(np.mean([r['argmax_equal'] for r in records])),
            'initial_adapter_sha256': initial, 'rng_restored': True, 'records': records}
    gate['verdict'] = 'PASS' if (len(records) == 84 and gate['max_abs_dlogscore'] < 1e-3
        and gate['max_abs_dlogp'] < 1e-3 and gate['max_kl'] < 1e-6 and gate['argmax_agreement'] == 1.0) else 'FAIL'
    write_new(args.output_dir / 'step0_gate.json', gate)
    if gate['verdict'] != 'PASS':
        raise ValueError(f"step-0 gate FAIL: { {k:v for k,v in gate.items() if k != 'records'} }")
    print('[preflight] step-0 gate PASS; training RNG restored', flush=True)
