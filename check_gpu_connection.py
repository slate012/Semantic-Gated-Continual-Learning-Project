#!/usr/bin/env python3
"""
ARC Labs GPU Connection Checker
================================

Run this BEFORE starting any training to verify:
1. SSH connection to ARC Labs workstation is alive
2. GPU (RTX 4090) is available and healthy
3. CUDA toolkit is working
4. Sufficient VRAM is free

Usage:
    python3 check_gpu_connection.py          # Check remote GPU
    python3 check_gpu_connection.py --local  # Check local GPU (if any)
"""

import subprocess
import sys
import os
import argparse
import json

# ── Colors ──
class C:
    G = '\033[92m'   # Green
    R = '\033[91m'   # Red
    Y = '\033[93m'   # Yellow
    B = '\033[94m'   # Blue
    BOLD = '\033[1m'
    END = '\033[0m'

SSH_HOST = "arcgpu"
SSH_TIMEOUT = 10  # seconds


def run_ssh_command(command: str, timeout: int = SSH_TIMEOUT) -> tuple:
    """Run a command on the remote GPU machine via SSH."""
    full_cmd = ["ssh", "-o", f"ConnectTimeout={timeout}", SSH_HOST, command]
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Connection timed out"
    except FileNotFoundError:
        return False, "", "SSH client not found"


def check_ssh_connection() -> bool:
    """Test basic SSH connectivity."""
    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  ARC Labs GPU Connection Check{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    print(f"  {C.B}[1/4]{C.END} Testing SSH connection to {C.BOLD}{SSH_HOST}{C.END}...")

    ok, out, err = run_ssh_command("echo CONNECTION_OK")

    if ok and "CONNECTION_OK" in out:
        print(f"  {C.G}✓ SSH connection successful{C.END}")
        return True
    else:
        print(f"  {C.R}✗ SSH connection FAILED{C.END}")
        if "timed out" in err.lower() or "timed out" in str(err).lower():
            print(f"    {C.Y}→ Are you on the campus WiFi/LAN?{C.END}")
            print(f"    {C.Y}→ ARC Labs is only accessible within the campus network{C.END}")
        elif "permission denied" in err.lower():
            print(f"    {C.Y}→ SSH key not accepted. Check that your public key was shared with ARC Labs{C.END}")
            print(f"    {C.Y}→ Key: ~/.ssh/id_ed25519.pub{C.END}")
        elif "host key" in err.lower():
            print(f"    {C.Y}→ Host key issue. Try: ssh-keygen -R '[10.3.32.62]:2222'{C.END}")
        else:
            print(f"    {C.Y}→ Error: {err}{C.END}")
        return False


def check_nvidia_smi() -> bool:
    """Check nvidia-smi output on remote."""
    print(f"\n  {C.B}[2/4]{C.END} Checking NVIDIA GPU via nvidia-smi...")

    ok, out, err = run_ssh_command("nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu --format=csv,noheader,nounits", timeout=15)

    if ok and out:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            gpu_name, mem_total, mem_free, driver, temp = parts[:5]
            print(f"  {C.G}✓ GPU detected:{C.END}")
            print(f"    GPU:          {C.BOLD}{gpu_name}{C.END}")
            print(f"    VRAM Total:   {mem_total} MB ({float(mem_total)/1024:.1f} GB)")
            print(f"    VRAM Free:    {mem_free} MB ({float(mem_free)/1024:.1f} GB)")
            print(f"    Driver:       {driver}")
            print(f"    Temperature:  {temp}°C")

            free_gb = float(mem_free) / 1024
            if free_gb < 8:
                print(f"    {C.Y}⚠ Low free VRAM ({free_gb:.1f} GB). Another user may be using the GPU.{C.END}")
            return True
        else:
            print(f"  {C.G}✓ GPU present but couldn't parse details:{C.END}")
            print(f"    {out}")
            return True
    else:
        print(f"  {C.R}✗ nvidia-smi failed{C.END}")
        print(f"    {C.Y}→ Error: {err or 'No output'}{C.END}")
        return False


def check_cuda_pytorch() -> bool:
    """Check CUDA availability via PyTorch on remote."""
    print(f"\n  {C.B}[3/4]{C.END} Checking PyTorch CUDA support...")

    python_check = """
import json, sys
try:
    import torch
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0
    }
    print(json.dumps(info))
except ImportError:
    print(json.dumps({"error": "pytorch_not_installed"}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"""

    ok, out, err = run_ssh_command(f'python3 -c "{python_check}"', timeout=30)

    if ok and out:
        try:
            info = json.loads(out)
            if "error" in info:
                if info["error"] == "pytorch_not_installed":
                    print(f"  {C.Y}⚠ PyTorch not installed on remote yet{C.END}")
                    print(f"    {C.Y}→ Will be installed by deploy_to_gpu.sh{C.END}")
                    return True  # Not a blocker
                else:
                    print(f"  {C.R}✗ Error: {info['error']}{C.END}")
                    return False

            if info["cuda_available"]:
                print(f"  {C.G}✓ PyTorch CUDA is working:{C.END}")
                print(f"    PyTorch:      {info['torch_version']}")
                print(f"    CUDA:         {info['cuda_version']}")
                print(f"    GPU:          {info['device_name']}")
                print(f"    VRAM:         {info['vram_gb']} GB")
                print(f"    GPU Count:    {info['device_count']}")
                return True
            else:
                print(f"  {C.R}✗ CUDA not available in PyTorch{C.END}")
                print(f"    PyTorch: {info['torch_version']}")
                return False
        except json.JSONDecodeError:
            print(f"  {C.Y}⚠ Could not parse response: {out}{C.END}")
            return True  # Non-blocking
    else:
        print(f"  {C.Y}⚠ Could not check PyTorch (may not be installed yet){C.END}")
        return True  # Not a blocker — deploy script will install it


def check_disk_space() -> bool:
    """Check available disk space on remote."""
    print(f"\n  {C.B}[4/4]{C.END} Checking disk space...")

    ok, out, err = run_ssh_command("df -h ~ | tail -1 | awk '{print $4}'")

    if ok and out:
        print(f"  {C.G}✓ Available disk space: {out}{C.END}")
        return True
    else:
        print(f"  {C.Y}⚠ Could not check disk space{C.END}")
        return True


def check_local_gpu():
    """Check local GPU (for reference)."""
    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  Local GPU Check{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  {C.G}✓ Local GPU: {name} ({vram:.1f} GB){C.END}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  {C.Y}ℹ Local device: Apple Silicon (MPS) — not suitable for this project{C.END}")
            print(f"    → Use the ARC Labs RTX 4090 instead")
        else:
            print(f"  {C.Y}ℹ No local GPU — use ARC Labs RTX 4090{C.END}")
    except ImportError:
        print(f"  {C.Y}ℹ PyTorch not installed locally{C.END}")


def print_summary(ssh_ok, gpu_ok, cuda_ok, disk_ok):
    """Print final summary."""
    all_ok = ssh_ok and gpu_ok

    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  Summary{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    checks = [
        ("SSH Connection", ssh_ok),
        ("NVIDIA GPU", gpu_ok),
        ("PyTorch CUDA", cuda_ok),
        ("Disk Space", disk_ok),
    ]

    for name, passed in checks:
        icon = f"{C.G}✓{C.END}" if passed else f"{C.R}✗{C.END}"
        print(f"  {icon} {name}")

    print()

    if all_ok:
        print(f"  {C.G}{C.BOLD}🚀 Ready to deploy! Run:{C.END}")
        print(f"     {C.B}bash deploy_to_gpu.sh{C.END}")
        print()
    else:
        print(f"  {C.R}{C.BOLD}❌ Not ready. Fix the issues above first.{C.END}")
        if not ssh_ok:
            print(f"     {C.Y}• Make sure you're on campus WiFi{C.END}")
            print(f"     {C.Y}• Try: ssh arcgpu{C.END}")
        print()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Check ARC Labs GPU connection")
    parser.add_argument("--local", action="store_true", help="Check local GPU instead")
    args = parser.parse_args()

    if args.local:
        check_local_gpu()
        return

    ssh_ok = check_ssh_connection()

    if not ssh_ok:
        print_summary(False, False, False, False)
        sys.exit(1)

    gpu_ok = check_nvidia_smi()
    cuda_ok = check_cuda_pytorch()
    disk_ok = check_disk_space()

    all_ok = print_summary(ssh_ok, gpu_ok, cuda_ok, disk_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
