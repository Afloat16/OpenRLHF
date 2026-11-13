import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "26494490bc237f6c3af62dab6117e6bbb0a78dba"
SOURCE = "3378c344009614dc0d8f49c2f0642abee96bb4bf"
PRODUCTION = "openrlhf/models/utils.py"
TEST = "tests/test_selective_log_probs_stability.py"
BENCHMARK = "contribution-validation/benchmark_log_probs.py"
ROOT = Path.cwd()


def run(command, name):
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    Path(f"{name}.log").write_text(completed.stdout)
    print(f"===== {name}: exit {completed.returncode} =====", flush=True)
    print(completed.stdout, flush=True)
    return completed.returncode


def test_totals(path):
    return {
        key: sum(int(suite.get(key, 0)) for suite in ET.parse(path).getroot().iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }


def benchmark(label):
    rows = []
    for n_rows, vocab in ((32, 512), (4097, 8192)):
        for grad in (False, True):
            command = [sys.executable, BENCHMARK, "--rows", str(n_rows), "--vocab", str(vocab)]
            if grad:
                command.append("--grad")
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            Path(f"{label}-{n_rows}-{int(grad)}-stderr.log").write_text(result.stderr)
            # DeepSpeed emits its accelerator banner before the benchmark record.
            rows.append(json.loads(result.stdout.splitlines()[-1]))
    Path(f"{label}-benchmark.json").write_text(json.dumps(rows, indent=2))
    print(
        label,
        [{key: value for key, value in row.items() if key not in ("log_probs", "gradient_samples")} for row in rows],
    )
    return rows


provenance = subprocess.run(
    [
        sys.executable,
        "-c",
        "import deepspeed, torch, transformers; print(deepspeed.__file__); "
        "assert deepspeed.__file__.endswith('deepspeed/__init__.py'); "
        "print(torch.__version__, transformers.__version__)",
    ],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=True,
)
print(provenance.stdout, flush=True)
Path("native-provenance.log").write_text(provenance.stdout)
print(
    "Dependency versions:",
    {name: importlib.metadata.version(name) for name in ("torch", "deepspeed", "transformers", "pytest")},
)
subprocess.run(["git", "diff", "--exit-code", SOURCE, "--", "openrlhf", "tests"], check=True)
subprocess.run(["git", "checkout", BASE, "--", PRODUCTION], check=True)
before_exit = run([sys.executable, "-m", "pytest", "-q", TEST, "--junitxml=baseline.xml"], "baseline")
before = test_totals("baseline.xml")
before_benchmark = benchmark("baseline")
subprocess.run(["git", "checkout", SOURCE, "--", PRODUCTION], check=True)
subprocess.run(["git", "diff", "--exit-code", SOURCE, "--", "openrlhf", "tests"], check=True)
after_exit = run([sys.executable, "-m", "pytest", "-q", TEST, "--junitxml=patched.xml"], "patched")
after = test_totals("patched.xml")
after_benchmark = benchmark("patched")
compile_exit = run([sys.executable, "-m", "compileall", "-q", "openrlhf", "examples", "tests"], "compileall")
# Run the repository's documented complete command separately. Optional distributed
# dependencies or pre-existing failures are reported, not confused with target tests.
full_exit = run([sys.executable, "-m", "pytest", "-q", "--junitxml=full-suite.xml"], "full-suite")
style_exit = 0
for command in (
    [sys.executable, "-m", "isort", "--check-only", "--diff", PRODUCTION, TEST],
    [sys.executable, "-m", "black", "--check", "--diff", PRODUCTION, TEST],
    [sys.executable, "-m", "autoflake", "--check-diff", "--remove-all-unused-imports", PRODUCTION, TEST],
):
    style_exit |= run(command, f"style-{command[2]}")
result = {
    "base": BASE,
    "source": SOURCE,
    "baseline": before,
    "patched": after,
    "baseline_exit": before_exit,
    "patched_exit": after_exit,
    "compileall_exit": compile_exit,
    "full_suite_exit": full_exit,
    "style_exit": style_exit,
    "scope": "Exact source module with real DeepSpeed and native Torch/HF GPT2; no FlashAttention, full Actor, GPU or distributed execution.",
}
Path("validation-result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
assert before_exit == 1 and before["failures"] > 0 and before["errors"] == 0, before
assert after_exit == 0 and after == {"tests": 44, "failures": 0, "errors": 0, "skipped": 0}, after
assert compile_exit == 0 and style_exit == 0
for original, patched in zip(before_benchmark, after_benchmark):
    assert (original["rows"], original["vocab"], original["grad"]) == (
        patched["rows"],
        patched["vocab"],
        patched["grad"],
    )
    for key in ("log_probs", "gradient_samples"):
        if original[key] is not None:
            assert len(original[key]) == len(patched[key])
            assert all(abs(a - b) <= 2e-6 for a, b in zip(original[key], patched[key])), key
