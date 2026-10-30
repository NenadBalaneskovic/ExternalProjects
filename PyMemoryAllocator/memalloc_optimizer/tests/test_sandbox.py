from pathlib import Path
from memalloc_core.execution_sandbox import ExecutionSandbox

sandbox = ExecutionSandbox()
result = sandbox.run_script(Path("D:/PyMemoryAllocator/memalloc_optimizer/examples/ranking_script.py"))

print("SUCCESS:", result.success)
print("RUNTIME:", result.runtime_seconds)
print("MEMORY:", result.peak_memory_mb)
print("STDOUT:", repr(result.stdout))
print("STDERR:", repr(result.stderr))
