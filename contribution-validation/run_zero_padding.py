import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "26494490bc237f6c3af62dab6117e6bbb0a78dba"
SOURCE = "2610aac7095e1e3476b00d3b2cb5f7027f355f33"
PRODUCTION = "openrlhf/utils/utils.py"
TEST = "tests/test_zero_pad_sequence_allocations.py"


def run(command, name):
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    Path(f"{name}.log").write_text(completed.stdout)
    print(f"===== {name}: exit {completed.returncode} =====", flush=True)
    print(completed.stdout, flush=True)
    return completed.returncode


def totals(path):
    return {
        key: sum(int(suite.get(key, 0)) for suite in ET.parse(path).getroot().iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }


benchmarks = {}
results = {}
for label, commit in (("baseline", BASE), ("patched", SOURCE)):
    subprocess.run(["git", "checkout", commit, "--", PRODUCTION], check=True)
    result = run([sys.executable, "-m", "pytest", "-q", TEST, f"--junitxml={label}.xml"], label)
    results[label] = {"exit": result, **totals(f"{label}.xml")}
    completed = subprocess.run(
        [sys.executable, "contribution-validation/benchmark_zero_padding.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    Path(f"{label}-benchmark.json").write_text(completed.stdout)
    Path(f"{label}-benchmark-stderr.log").write_text(completed.stderr)
    benchmarks[label] = json.loads(completed.stdout)
    print(completed.stdout, flush=True)
subprocess.run(["git", "diff", "--exit-code", SOURCE, "--", "openrlhf", "tests"], check=True)
results["compileall_exit"] = run(
    [sys.executable, "-m", "compileall", "-q", "openrlhf", "examples", "tests"], "compileall"
)
results["full_suite_exit"] = run([sys.executable, "-m", "pytest", "-q", "--junitxml=full-suite.xml"], "full-suite")
results["style_exit"] = 0
for command in (
    [sys.executable, "-m", "isort", "--check-only", "--diff", PRODUCTION, TEST],
    [sys.executable, "-m", "black", "--check", "--diff", PRODUCTION, TEST],
    [sys.executable, "-m", "autoflake", "--check-diff", "--remove-all-unused-imports", PRODUCTION, TEST],
):
    results["style_exit"] |= run(command, f"style-{command[2]}")
results.update(
    base=BASE,
    source=SOURCE,
    scope="Native CPU Torch, public SFTDataset/DataLoader, local HF GPT2 three-step updates; no GPU or distributed run.",
)
Path("validation-result.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2), flush=True)
assert results["baseline"] == {"exit": 1, "tests": 46, "failures": 2, "errors": 0, "skipped": 0}, results
assert results["patched"] == {"exit": 0, "tests": 46, "failures": 0, "errors": 0, "skipped": 0}, results
assert results["compileall_exit"] == 0 and results["style_exit"] == 0, results
for before, after in zip(benchmarks["baseline"]["rows"], benchmarks["patched"]["rows"]):
    assert before["output_sha256"] == after["output_sha256"]
    assert after["operator_allocated_bytes"] < before["operator_allocated_bytes"]
