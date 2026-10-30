"""
ULTRA-FAST synthetic benchmark for MemAlloc Optimizer.

This version:
- Does NOT use MemAllocController
- Does NOT run static analysis
- Does NOT run optimization plan
- Does NOT run codegen
- Does NOT run optimized execution
- Does NOT generate plots
- Does NOT spawn subprocesses
- Executes workloads directly in-process

Runtime: ~0.05–0.2 seconds total.
"""

import tempfile
from pathlib import Path
import time
import runpy

WORKLOADS = {
    "temporary_array": """
import numpy as np
def run():
    x = np.zeros(500000)
    return x.sum()
""",

    "repeated_allocation": """
import numpy as np
def run():
    total = 0
    for i in range(200):
        x = np.zeros(20000)
        total += x.sum()
    return total
""",

    "nested_loop": """
def run():
    s = 0
    for i in range(300):
        for j in range(300):
            s += (i * j) % 7
    return s
""",

    "large_allocation": """
import numpy as np
def run():
    x = np.zeros(20_000_000)
    return x.mean()
""",

    "mixed_pattern": """
import numpy as np
def run():
    total = 0
    for i in range(50):
        x = np.zeros(50000)
        y = np.zeros(100000)
        total += x.sum() + y.sum()
    for i in range(200):
        for j in range(200):
            total += (i * j) % 5
    return total
""",
}


def run_fast(name, code):
    print(f"\n=== FAST workload: {name} ===")

    # Write workload to temp file
    script_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".py").name)
    script_path.write_text(code)

    # Execute directly in-process (no sandbox)
    start = time.perf_counter()
    module_globals = runpy.run_path(str(script_path))
    result = module_globals["run"]()
    end = time.perf_counter()

    print(f"Result: {result}")
    print(f"Runtime: {end - start:.6f} seconds")
    print(f"=== Finished {name} ===")


def main():
    for name, code in WORKLOADS.items():
        run_fast(name, code)


if __name__ == "__main__":
    main()
