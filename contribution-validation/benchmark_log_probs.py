import argparse
import importlib.util
import json
import resource
import statistics
import time
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--vocab", type=int, required=True)
parser.add_argument("--grad", action="store_true")
args = parser.parse_args()
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("native_utils", Path("openrlhf/models/utils.py").resolve())
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
generator = torch.Generator().manual_seed(42)
logits = torch.randn(args.rows, args.vocab, generator=generator, requires_grad=args.grad)
labels = torch.arange(args.rows) % args.vocab
input_pointers = {logits.untyped_storage().data_ptr(), labels.untyped_storage().data_ptr()}
saved = {}


def pack(tensor):
    storage = tensor.untyped_storage()
    if storage.data_ptr() not in input_pointers:
        saved[storage.data_ptr()] = storage.nbytes()
    return tensor


rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with torch.set_grad_enabled(args.grad), torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
    result = module.log_probs_from_logits(logits, labels)
loss = result.mean()
if args.grad:
    loss.backward()
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
observed = result.detach().tolist()
grad_samples = logits.grad.flatten()[:: max(1, logits.numel() // 64)].tolist() if args.grad else None
saved_bytes = sum(saved.values())
del result, loss
durations = []
for repeat in range(6):
    logits.grad = None
    start = time.perf_counter()
    with torch.set_grad_enabled(args.grad):
        result = module.log_probs_from_logits(logits, labels)
        if args.grad:
            result.mean().backward()
    elapsed = time.perf_counter() - start
    if repeat:
        durations.append(elapsed)
    del result
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "rows": args.rows,
            "vocab": args.vocab,
            "grad": args.grad,
            "extra_saved_storage_bytes": saved_bytes,
            "first_step_peak_rss_kib": rss_after,
            "before_step_peak_rss_kib": rss_before,
            "median_seconds": statistics.median(durations),
            "all_seconds": durations,
            "log_probs": observed,
            "gradient_samples": grad_samples,
        }
    )
)
