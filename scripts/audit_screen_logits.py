"""Train-only, supplied-history audit of parallel versus frontier steering.

No generation, fitting, candidate selection, or held-out evaluation occurs here.
The parallel intervention is the actual production likelihood scorer; a recording
adapter retains its logits without changing the forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from safety_governor.models import (
    SteeringSite, _governor_intervention, _prefix_ids,
    load_transformerlens_model, response_negative_log_likelihood_governed,
    response_negative_log_likelihood_frontier,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directions', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-frontier-scorer', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads((args.directions / 'manifest.json').read_text())
    if manifest['dataset_sha256'] != sha(args.dataset):
        raise ValueError('direction-fitting dataset hash mismatch')
    sites, hashes = [], {}
    for layer in (12, 16, 20, 24):
        root = args.directions / 'directions' / f'layer_{layer:02d}'
        metadata = json.loads((root / 'metadata.json').read_text())
        vector_path = root / 'balanced_ridge.npy'
        expected = metadata['directions']['balanced_ridge']['sha256']
        if metadata['fit_split'] != 'train' or sha(vector_path) != expected:
            raise ValueError(f'layer {layer}: train provenance/hash mismatch')
        hashes[str(layer)] = expected
        sites.append(SteeringSite(layer, np.load(vector_path, allow_pickle=False), .25))
    selected = {}
    for line in args.dataset.read_text().splitlines():
        row = json.loads(line)
        if row['split'] == 'train' and row['polarity'] == 'unsafe':
            selected.setdefault(row['archetype'], row)
    if len(selected) != 4:
        raise ValueError('expected exactly four train archetypes')
    print('Verified frozen vectors; loading cached pinned model', flush=True)
    revision = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
    model = load_transformerlens_model(
        'meta-llama/Meta-Llama-3-8B-Instruct', revision, 'cuda', 'bfloat16')
    model.eval()
    device = next(model.parameters()).device

    class Recorder:
        def __getattr__(self, name):
            return getattr(model, name)

        def run_with_hooks(self, *a, **kw):
            result = model.run_with_hooks(*a, **kw)
            self.logits = result.detach()
            return result

    recorder = Recorder()
    output = {
        'scope': 'four_train_unsafe_supplied_histories_first_16_predictors',
        'model_revision': revision, 'dtype': 'bfloat16',
        'budget': .20, 'site_weights': [.25] * 4,
        'vector_sha256': hashes, 'dataset_sha256': sha(args.dataset),
        'direction_manifest_sha256': sha(args.directions / 'manifest.json'),
        'script_sha256': sha(__file__),
        'models_py_sha256': sha(Path(__file__).parents[1] / 'safety_governor/models.py'),
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'gpu': torch.cuda.get_device_name(),
        'packages': {p: importlib.metadata.version(p) for p in
                     ('torch', 'transformers', 'transformer-lens', 'numpy')},
        'attention_implementation': getattr(model.original_model.config, '_attn_implementation', None),
        'confirmatory_accessed': False, 'results': [],
    }
    with torch.inference_mode():
        for archetype, row in sorted(selected.items()):
            prefix = _prefix_ids(model.tokenizer, row['instruction'])
            response = list(model.tokenizer.encode(row['completion'], add_special_tokens=False))
            count = len(response) if args.verify_frontier_scorer else min(16, len(response))
            if count < 2:
                raise ValueError('need at least two supplied tokens')
            tokens = torch.tensor([prefix + response], device=device)
            boundary = len(prefix) - 1
            baseline = model.run_with_hooks(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool),
                                            return_type='logits', fwd_hooks=[])
            baseline = baseline[0, boundary:boundary + count].float().cpu()
            loss, total = response_negative_log_likelihood_governed(
                recorder, row['instruction'], row['completion'], sites, total_relative_l2=.20)
            parallel = recorder.logits[0, boundary:boundary + count].float().cpu()
            del recorder.logits
            per_token = []
            reference_base_loss = 0.0
            reference_steered_loss = 0.0
            for i in range(count):
                history = tokens[:, :len(prefix) + i]
                mask = torch.ones_like(history, dtype=torch.bool)
                common = dict(attention_mask=mask, return_type='logits')
                base_device = model.run_with_hooks(history, fwd_hooks=[], **common)[0, -1].float()
                base = base_device.cpu()
                hooks = [(f'blocks.{s.layer}.hook_resid_pre',
                          _governor_intervention(s, history.shape[1] - 1, .20, None)) for s in sites]
                frontier_device = model.run_with_hooks(history, fwd_hooks=hooks, **common)[0, -1].float()
                frontier = frontier_device.cpu()
                target = response[i]
                # Match the scorer's float32 CUDA loss kernel, not a CPU
                # recomputation with a different logsumexp reduction.
                target_device = torch.tensor([target], device=device)
                reference_base_loss += float(torch.nn.functional.cross_entropy(base_device.unsqueeze(0), target_device).cpu())
                reference_steered_loss += float(torch.nn.functional.cross_entropy(frontier_device.unsqueeze(0), target_device).cpu())
                delta = parallel[i] - frontier
                floor = baseline[i] - base
                per_token.append({
                    'index': i, 'target_id': target,
                    'baseline_max_abs_logit_gap': float(floor.abs().max()),
                    'steered_max_abs_logit_gap': float(delta.abs().max()),
                    'baseline_rms_logit_gap': float(floor.square().mean().sqrt()),
                    'steered_rms_logit_gap': float(delta.square().mean().sqrt()),
                    'steered_argmax_agrees': bool(parallel[i].argmax() == frontier.argmax()),
                    'parallel_target_logprob': float(parallel[i].log_softmax(-1)[target]),
                    'frontier_target_logprob': float(frontier.log_softmax(-1)[target]),
                })
            verification = {}
            if args.verify_frontier_scorer:
                actual_base, base_count = response_negative_log_likelihood_frontier(
                    model, row['instruction'], row['completion'])
                actual_steered, steered_count = response_negative_log_likelihood_frontier(
                    model, row['instruction'], row['completion'], sites, total_relative_l2=.20)
                verification = dict(baseline_nll=actual_base, reference_baseline_nll=reference_base_loss,
                    steered_nll=actual_steered, reference_steered_nll=reference_steered_loss,
                    baseline_abs_error=abs(actual_base-reference_base_loss),
                    steered_abs_error=abs(actual_steered-reference_steered_loss),
                    passed=(base_count == steered_count == count and
                            abs(actual_base-reference_base_loss) < 1e-4 and
                            abs(actual_steered-reference_steered_loss) < 1e-4))
                output['scope'] = 'four_train_unsafe_full_supplied_histories_frontier_scorer_verification'
            output['results'].append(dict(archetype=archetype, pair_id=row['pair_id'],
                split='train', polarity='unsafe', prefix_tokens=len(prefix),
                full_completion_tokens=total, full_parallel_nll_sum=loss,
                frontier_verification=verification, per_token=per_token))
            output['elapsed_seconds'] = time.monotonic() - started
            (args.output / 'audit.json').write_text(json.dumps(output, indent=2) + '\n')
            print(f'{archetype}: {count} supplied positions audited', flush=True)
    print(f'Complete: {args.output / "audit.json"}', flush=True)
    if args.verify_frontier_scorer and not all(r['frontier_verification']['passed'] for r in output['results']):
        raise RuntimeError('frontier scorer verification failed')


if __name__ == '__main__':
    main()
