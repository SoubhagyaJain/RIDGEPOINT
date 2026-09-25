"""Print a machine manifest and prove a small BF16 CUDA operation works."""

import json
import platform
import subprocess
import sys

import torch


def main() -> None:
    report = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,"
             "temperature.gpu,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        report["nvidia_smi"] = smi.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        report["nvidia_smi_error"] = str(exc)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        report["gpu"] = torch.cuda.get_device_name(device)
        report["capability"] = torch.cuda.get_device_capability(device)
        report["mem_free_bytes"], report["mem_total_bytes"] = torch.cuda.mem_get_info(device)
        x = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize(device)
        report["bf16_matmul_result"] = float(y[0, 0].item())
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
    print(json.dumps(report, indent=2))
    if not report["cuda_available"] or report.get("bf16_matmul_result") != 16.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
