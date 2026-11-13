import hashlib
import json
import statistics
import time

import torch

from openrlhf.trainer.ppo_utils import Experience, apply_overlong_penalty, apply_stop_properly_penalty

torch.set_num_threads(1)
rows = []
for size in (1, 64, 4096, 32768):
    lengths = torch.arange(size) % 8193
    flags = torch.arange(size) % 3 == 0
    initial = ((torch.arange(size) % 31).float() - 15) / 16

    def run():
        experience = Experience(rewards=initial.clone(), response_length=lengths, truncated=flags)
        started = time.perf_counter()
        penalized = apply_overlong_penalty([experience], 8192, 4096, 0.3)
        truncated = apply_stop_properly_penalty([experience], 0.5)
        elapsed = time.perf_counter() - started
        return elapsed, experience.rewards, penalized, truncated

    run()
    samples = [run() for _ in range(5)]
    elapsed, rewards, penalized, truncated = samples[-1]
    rows.append(
        {
            "batch_size": size,
            "median_seconds": statistics.median(sample[0] for sample in samples),
            "samples_seconds": [sample[0] for sample in samples],
            "penalized_count": penalized,
            "truncated_count": truncated,
            "reward_sha256": hashlib.sha256(rewards.numpy().tobytes()).hexdigest(),
        }
    )
print(json.dumps({"torch": torch.__version__, "cpu_threads": torch.get_num_threads(), "rows": rows}, indent=2))
