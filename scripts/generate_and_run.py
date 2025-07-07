import sys
import os
import torch
import numpy as np
from safetensors.torch import save_file
from safetensors import safe_open
from pathlib import Path

if (len(sys.argv) != 2):
    print("Usage: python3 ./generate_data.py <N>")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
GEN_DATA_FILE = DATA_DIR / "matmul_data.safetensors"
RESULT_FILE = DATA_DIR / "output.safetensors"


N = int(sys.argv[1])

A = torch.randn(N,N)
B = torch.randn(N,N)
C = torch.matmul(A, B)

save_file({
    "A": A,
    "B": B,
    "C": C
}, str(GEN_DATA_FILE))


import subprocess

rust_bin = "../target/release/volgaray_gemm"
env = os.environ.copy()
env["RUST_LOG"]="info"

with subprocess.Popen(
    [rust_bin, "-i", GEN_DATA_FILE, "-o", RESULT_FILE],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
) as process:
    for line in process.stdout:
        print(line.strip())
    for line in process.stderr:
        print(line.strip(), file=sys.stderr)


with safe_open(RESULT_FILE, framework="pt") as f:
    gemm_result = f.get_tensor("C")



mse = torch.mean((gemm_result - C) ** 2)
rmse = torch.sqrt(mse)

print("RMSE:", rmse)
