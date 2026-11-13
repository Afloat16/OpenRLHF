import hashlib
import json
import statistics
import time

import torch

from openrlhf.utils.utils import zero_pad_sequences

torch.set_num_threads(2)
rows = []
for batch_size, length, ragged in ((1, 8192, False), (8, 8192, False), (32, 8192, False), (32, 8192, True)):
    inputs = [
        torch.arange(length - (i % 8 if ragged else 0), dtype=torch.int64).unsqueeze(0) for i in range(batch_size)
    ]
    output = zero_pad_sequences(inputs, side="right")
    expected = output.clone()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        output = zero_pad_sequences(inputs, side="right")
    allocated = sum(max(event.self_cpu_memory_usage, 0) for event in profile.events())
    durations = []
    for repeat in range(8):
        start = time.perf_counter()
        for _ in range(100):
            output = zero_pad_sequences(inputs, side="right")
        elapsed = (time.perf_counter() - start) / 100
        if repeat:
            durations.append(elapsed)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    rows.append(
        {
            "batch_size": batch_size,
            "length": length,
            "ragged": ragged,
            "output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
            "operator_allocated_bytes": allocated,
            "output_bytes": output.numel() * output.element_size(),
            "median_seconds": statistics.median(durations),
            "all_seconds": durations,
        }
    )
print(json.dumps({"torch": torch.__version__, "threads": torch.get_num_threads(), "rows": rows}, indent=2))
