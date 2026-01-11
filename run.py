#!/usr/bin/env python3
"""
WakeBuilder Setup and Run Script

A production-level automation script for setting up and running WakeBuilder.
This script handles:
- Python version verification (3.12+)
- Package manager detection (uv preferred, pip fallback)
- Virtual environment creation and activation
- Dependency installation
- TTS voice and model downloading
- Server startup

Usage:
    python run.py              # Full setup and run
    python run.py --check      # Verify environment only
    python run.py --install    # Install dependencies only
    python run.py --download   # Download TTS models only
    python run.py --run        # Run server only (skip checks)
    python run.py --help       # Show help

This script is designed to work:
- On first-time project setup
- For verification of existing installations
- Cross-platform (Windows, Linux, macOS)
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
VENV_DIR = SCRIPT_DIR / ".venv"
MIN_PYTHON_VERSION = (3, 12)
REQUIRED_FREE_MEMORY_GB = 8
RECOMMENDED_FREE_MEMORY_GB = 16

# PyTorch CUDA index URLs
PYTORCH_CUDA_INDEXES = {
    "12.8": "https://download.pytorch.org/whl/cu128",
    "12.6": "https://download.pytorch.org/whl/cu126",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
DEFAULT_CUDA_VERSION = "12.8"

# Directories to check/create
REQUIRED_DIRS = ["models", "data", "tts_voices", "recordings"]

# =============================================================================
# Terminal Styling (works without external dependencies)
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    # Check if terminal supports colors
    ENABLED = (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("TERM", "") != "dumb"
        and os.environ.get("NO_COLOR") is None
    )

    # Colors (only applied if terminal supports them)
    RESET = "\033[0m" if ENABLED else ""
    BOLD = "\033[1m" if ENABLED else ""
    DIM = "\033[2m" if ENABLED else ""

    # Foreground colors
    RED = "\033[91m" if ENABLED else ""
    GREEN = "\033[92m" if ENABLED else ""
    YELLOW = "\033[93m" if ENABLED else ""
    BLUE = "\033[94m" if ENABLED else ""
    MAGENTA = "\033[95m" if ENABLED else ""
    CYAN = "\033[96m" if ENABLED else ""
    WHITE = "\033[97m" if ENABLED else ""

    # Background colors
    BG_RED = "\033[41m" if ENABLED else ""
    BG_GREEN = "\033[42m" if ENABLED else ""
    BG_YELLOW = "\033[43m" if ENABLED else ""
    BG_BLUE = "\033[44m" if ENABLED else ""


class Symbols:
    """Unicode symbols with ASCII fallbacks."""

    # Try to use Unicode, fall back to ASCII
    try:
        "✓✗→•".encode(sys.stdout.encoding or "utf-8")
        CHECK = "✓"
        CROSS = "✗"
        ARROW = "→"
        BULLET = "•"
        SPARKLE = "✨"
        ROCKET = "🚀"
        WARN = "⚠️"
        INFO = "ℹ️"
        GEAR = "⚙️"
        PACKAGE = "📦"
        FOLDER = "📁"
        DOWNLOAD = "⬇️"
        SUCCESS = "✅"
        FAIL = "❌"
    except (UnicodeEncodeError, LookupError):
        CHECK = "[OK]"
        CROSS = "[X]"
        ARROW = "->"
        BULLET = "*"
        SPARKLE = "*"
        ROCKET = ">"
        WARN = "[!]"
        INFO = "[i]"
        GEAR = "[*]"
        PACKAGE = "[P]"
        FOLDER = "[D]"
        DOWNLOAD = "[D]"
        SUCCESS = "[OK]"
        FAIL = "[FAIL]"


def print_header(text: str) -> None:
    """Print a styled header."""
    width = 70
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print()


def print_section(text: str) -> None:
    """Print a section header."""
    print()
    print(f"{Colors.BLUE}{Colors.BOLD}{Symbols.ARROW} {text}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 50}{Colors.RESET}")


def print_step(text: str) -> None:
    """Print a step description."""
    print(f"  {Colors.WHITE}{Symbols.BULLET} {text}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"  {Colors.GREEN}{Symbols.CHECK} {text}{Colors.RESET}")


def print_fail(text: str) -> None:
    """Print a failure message."""
    print(f"  {Colors.RED}{Symbols.CROSS} {text}{Colors.RESET}")


def print_warn(text: str) -> None:
    """Print a warning message."""
    print(f"  {Colors.YELLOW}{Symbols.WARN} {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"  {Colors.CYAN}{Symbols.INFO} {text}{Colors.RESET}")


def prompt_yn(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        question: The question to ask
        default: Default value if user just presses Enter

    Returns:
        True for yes, False for no
    """
    if default:
        prompt = f"  {Colors.YELLOW}? {question} [Y/n]: {Colors.RESET}"
    else:
        prompt = f"  {Colors.YELLOW}? {question} [y/N]: {Colors.RESET}"

    try:
        response = input(prompt).strip().lower()
        if not response:
            return default
        return response in ("y", "yes", "ye", "1", "true")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_banner() -> None:
    """Print the WakeBuilder banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ██╗    ██╗ █████╗ ██╗  ██╗███████╗██████╗ ██╗   ██╗██╗██╗     ██████╗ ███████╗██████╗ 
    ██║    ██║██╔══██╗██║ ██╔╝██╔════╝██╔══██╗██║   ██║██║██║     ██╔══██╗██╔════╝██╔══██╗
    ██║ █╗ ██║███████║█████╔╝ █████╗  ██████╔╝██║   ██║██║██║     ██║  ██║█████╗  ██████╔╝
    ██║███╗██║██╔══██║██╔═██╗ ██╔══╝  ██╔══██╗██║   ██║██║██║     ██║  ██║██╔══╝  ██╔══██╗
    ╚███╔███╔╝██║  ██║██║  ██╗███████╗██████╔╝╚██████╔╝██║███████╗██████╔╝███████╗██║  ██║
     ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝
{Colors.RESET}
{Colors.DIM}    Custom Wake Word Detection - Setup & Run Script{Colors.RESET}
    """
    print(banner)


# =============================================================================
# System Checks
# =============================================================================


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    current = sys.version_info[:2]
    required = MIN_PYTHON_VERSION

    if current >= required:
        return (
            True,
            f"Python {current[0]}.{current[1]} (required: {required[0]}.{required[1]}+)",
        )
    else:
        return (
            False,
            f"Python {current[0]}.{current[1]} is too old (required: {required[0]}.{required[1]}+)",
        )


def check_in_venv() -> bool:
    """Check if currently running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
        or os.environ.get("VIRTUAL_ENV") is not None  # explicit env var
    )


def find_package_manager() -> Tuple[str, str]:
    """Find available package manager (uv preferred, pip fallback)."""
    # Check for uv
    uv_path = shutil.which("uv")
    if uv_path:
        return "uv", uv_path

    # Check for pip
    pip_path = shutil.which("pip") or shutil.which("pip3")
    if pip_path:
        return "pip", pip_path

    return "none", ""


def detect_cuda_version() -> Optional[str]:
    """Detect installed CUDA version using nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        # Get CUDA version from nvidia-smi
        result2 = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=10
        )

        # Parse CUDA version from nvidia-smi output
        # Look for "CUDA Version: X.Y"
        for line in result2.stdout.split("\n"):
            if "CUDA Version" in line:
                # Extract version number
                import re

                match = re.search(r"CUDA Version:\s*(\d+\.\d+)", line)
                if match:
                    return match.group(1)

        return None
    except Exception:
        return None


def get_pytorch_index(cuda_version: Optional[str]) -> Tuple[str, str]:
    """Get PyTorch index URL for the given CUDA version."""
    if cuda_version is None or cuda_version.lower() == "cpu":
        return PYTORCH_CUDA_INDEXES["cpu"], "CPU"

    # Match to closest available version
    cuda_major_minor = cuda_version.split(".")[:2]
    cuda_key = ".".join(cuda_major_minor)

    if cuda_key in PYTORCH_CUDA_INDEXES:
        return PYTORCH_CUDA_INDEXES[cuda_key], f"CUDA {cuda_key}"

    # Try to match major version
    cuda_major = cuda_major_minor[0]
    for key in sorted(PYTORCH_CUDA_INDEXES.keys(), reverse=True):
        if key.startswith(cuda_major + "."):
            return PYTORCH_CUDA_INDEXES[key], f"CUDA {key}"

    # Default to latest
    return (
        PYTORCH_CUDA_INDEXES[DEFAULT_CUDA_VERSION],
        f"CUDA {DEFAULT_CUDA_VERSION} (default)",
    )


def get_venv_python() -> Path:
    """Get the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip() -> Path:
    """Get the path to pip in the virtual environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def get_venv_uvicorn() -> Path:
    """Get the path to uvicorn in the virtual environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "uvicorn.exe"
    return VENV_DIR / "bin" / "uvicorn"


def run_command(
    cmd: list,
    cwd: Optional[Path] = None,
    capture: bool = False,
    env: Optional[dict] = None,
) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or SCRIPT_DIR,
            capture_output=capture,
            text=True,
            env={**os.environ, **(env or {})},
        )
        stdout = result.stdout if capture else ""
        stderr = result.stderr if capture else ""
        return result.returncode, stdout, stderr
    except Exception as e:
        return 1, "", str(e)


# =============================================================================
# Setup Functions
# =============================================================================


def create_directories(step_num: int = 4) -> bool:
    """Create required directories if they don't exist."""
    print_header(f"Step {step_num}: Creating Directories")
    all_ok = True

    for dir_name in REQUIRED_DIRS:
        dir_path = SCRIPT_DIR / dir_name
        if dir_path.exists():
            print_success(f"{dir_name}/ exists")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print_success(f"{dir_name}/ created")
            except Exception as e:
                print_fail(f"Failed to create {dir_name}/: {e}")
                all_ok = False

    return all_ok


def create_venv(pkg_manager: str, step_num: int = 3) -> bool:
    """Create virtual environment if it doesn't exist."""
    print_header(f"Step {step_num}: Virtual Environment")

    if VENV_DIR.exists() and get_venv_python().exists():
        print_success(f"Virtual environment exists at {VENV_DIR}")
        return True

    print_step("Creating virtual environment...")

    if pkg_manager == "uv":
        # Use uv to create venv
        code, _, stderr = run_command(["uv", "venv", str(VENV_DIR)])
    else:
        # Use built-in venv module
        code, _, stderr = run_command([sys.executable, "-m", "venv", str(VENV_DIR)])

    if code == 0 and get_venv_python().exists():
        print_success(f"Created virtual environment at {VENV_DIR}")
        return True
    else:
        print_fail(f"Failed to create virtual environment: {stderr}")
        return False


def install_dependencies(
    pkg_manager: str, pytorch_index: str, pytorch_label: str, step_num: int = 5
) -> bool:
    """Install project dependencies."""
    print_header(f"Step {step_num}: Installing Dependencies")

    venv_python = get_venv_python()
    venv_pip = get_venv_pip()

    if not venv_python.exists():
        print_fail("Virtual environment not found. Run with --install first.")
        return False

    pyproject = SCRIPT_DIR / "pyproject.toml"
    if not pyproject.exists():
        print_fail("pyproject.toml not found")
        return False

    # Check if dependencies are already installed
    print_step("Checking existing dependencies...")
    code, stdout, _ = run_command(
        [
            str(venv_python),
            "-c",
            'import uvicorn; import torch; import transformers; print("OK")',
        ],
        capture=True,
    )

    if code == 0 and "OK" in stdout:
        # Check if PyTorch has CUDA support
        code2, stdout2, _ = run_command(
            [
                str(venv_python),
                "-c",
                'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")',
            ],
            capture=True,
        )
        cuda_status = "CUDA" if "CUDA" in stdout2 else "CPU-only"
        print_success(f"Core dependencies already installed ({cuda_status})")
        return True

    print_step("Installing dependencies (this may take several minutes)...")

    # Step 1: Install PyTorch with appropriate CUDA/CPU support
    print_info(f"Installing PyTorch ({pytorch_label})...")

    if pkg_manager == "uv":
        # Use uv pip install with CUDA index
        code, _, stderr = run_command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                pytorch_index,
            ]
        )
    else:
        # Use pip with CUDA index
        code, _, stderr = run_command(
            [
                str(venv_pip),
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                pytorch_index,
            ]
        )

    if code != 0:
        print_warn(
            f"PyTorch CUDA installation warning: {stderr[:100] if stderr else 'unknown'}"
        )
        print_info("Falling back to default PyTorch (may be CPU-only)...")
    else:
        print_success("PyTorch with CUDA installed")

    # Step 2: Install remaining dependencies
    print_step("Installing remaining dependencies...")

    if pkg_manager == "uv":
        # Use uv sync for remaining packages
        code, _, stderr = run_command(["uv", "sync"])
        if code != 0:
            print_warn(f"uv sync warning: {stderr[:100] if stderr else 'unknown'}")
            # Fallback to uv pip install
            code, _, stderr = run_command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    "-r",
                    "pyproject.toml",
                ]
            )
    else:
        # Use pip
        code, _, stderr = run_command([str(venv_pip), "install", "-e", "."])

    if code == 0:
        print_success("Dependencies installed successfully")
        return True
    else:
        print_fail(
            f"Dependency installation failed: {stderr[:100] if stderr else 'unknown'}"
        )
        return False


def download_tts_models(step_num: int = 6) -> bool:
    """Download TTS voices and models."""
    print_header(f"Step {step_num}: Downloading TTS Models")

    venv_python = get_venv_python()
    if not venv_python.exists():
        print_fail("Virtual environment not found")
        return False

    # Check if voices already exist
    voices_dir = SCRIPT_DIR / "tts_voices"
    Path.home() / ".cache" / "huggingface"

    piper_voices = list(voices_dir.glob("*.onnx")) if voices_dir.exists() else []

    # Check for at least 80 of 87 expected voices
    if len(piper_voices) >= 80:
        print_success(
            f"Piper TTS voices already downloaded ({len(piper_voices)} voices)"
        )
    else:
        print_step(f"Downloading Piper TTS voices ({len(piper_voices)}/87 present)...")
        download_script = SCRIPT_DIR / "scripts" / "download_voices.py"
        if download_script.exists():
            code, _, stderr = run_command([str(venv_python), str(download_script)])
            if code == 0:
                print_success("Piper voices downloaded (87 voices)")
            else:
                print_warn(
                    f"Some Piper voices may have failed: {stderr[:100] if stderr else 'check logs'}"
                )
        else:
            print_warn("download_voices.py not found, skipping Piper voices")

    # Check for cached models
    print_step("Checking TTS model cache...")
    preload_script = SCRIPT_DIR / "scripts" / "preload_tts_models.py"

    if preload_script.exists():
        # Quick check if models are cached
        code, stdout, _ = run_command(
            [
                str(venv_python),
                "-c",
                "from pathlib import Path; "
                'cache = Path.home() / ".cache" / "huggingface"; '
                'print("CACHED" if (cache / "hub").exists() and len(list((cache / "hub").glob("*"))) > 5 else "MISSING")',
            ],
            capture=True,
        )

        if "CACHED" in stdout:
            print_success("TTS models already cached")
        else:
            print_step("Pre-loading TTS models (this takes a while on first run)...")
            print_info("Models: Coqui TTS, Kokoro TTS")
            code, _, _ = run_command([str(venv_python), str(preload_script)])
            if code == 0:
                print_success("TTS models pre-loaded")
            else:
                print_warn("Some TTS models may have failed to preload")
    else:
        print_warn("preload_tts_models.py not found, skipping model preload")

    return True


def verify_installation(step_num: int = 7) -> bool:
    """Verify that the installation is complete and working."""
    print_header(f"Step {step_num}: Verifying Installation")

    venv_python = get_venv_python()
    if not venv_python.exists():
        print_fail("Virtual environment not found")
        return False

    checks = [
        (
            "Python version",
            f"import sys; assert sys.version_info >= {MIN_PYTHON_VERSION}",
        ),
        ("PyTorch", "import torch; print(f'PyTorch {torch.__version__}')"),
        ("CUDA available", "import torch; print(f'CUDA: {torch.cuda.is_available()}')"),
        ("Transformers", "import transformers"),
        ("FastAPI", "import fastapi"),
        ("Uvicorn", "import uvicorn"),
        ("WakeBuilder", "from src.wakebuilder.backend.main import app"),
    ]

    all_ok = True
    for name, check_code in checks:
        code, stdout, stderr = run_command(
            [str(venv_python), "-c", check_code], capture=True
        )
        if code == 0:
            output = stdout.strip() if stdout.strip() else ""
            msg = f"{name}" + (f" - {output}" if output else "")
            print_success(msg)
        else:
            print_fail(f"{name} - {stderr[:50]}")
            all_ok = False

    return all_ok


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the uvicorn server."""
    print_section("Starting WakeBuilder Server")

    venv_uvicorn = get_venv_uvicorn()
    venv_python = get_venv_python()

    if venv_uvicorn.exists():
        cmd = [str(venv_uvicorn)]
    else:
        cmd = [str(venv_python), "-m", "uvicorn"]

    cmd.extend(
        ["src.wakebuilder.backend.main:app", "--host", host, "--port", str(port)]
    )

    print_info(f"Server starting at http://{host}:{port}")
    print_info("Press Ctrl+C to stop")
    print()

    # Run uvicorn (this blocks until Ctrl+C)
    os.chdir(SCRIPT_DIR)
    try:
        subprocess.run(cmd, cwd=SCRIPT_DIR)
    except KeyboardInterrupt:
        print()
        print_info("Server stopped")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    cuda_versions = ", ".join(PYTORCH_CUDA_INDEXES.keys())

    parser = argparse.ArgumentParser(
        description="WakeBuilder Setup and Run Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python run.py              Interactive setup (prompts for each step)
  python run.py --auto       Automatic setup (no prompts)
  python run.py --cuda 12.4  Use specific CUDA version
  python run.py --cuda cpu   Install CPU-only PyTorch
  python run.py --check      Verify environment only
  python run.py --install    Install dependencies only
  python run.py --download   Download TTS models only
  python run.py --run        Run server (skip setup)

Available CUDA versions: {cuda_versions}
        """,
    )
    parser.add_argument(
        "--auto",
        "-y",
        action="store_true",
        help="Automatic mode (no prompts, answer yes to all)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify environment and dependencies only"
    )
    parser.add_argument(
        "--install", action="store_true", help="Install dependencies only"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download TTS models only"
    )
    parser.add_argument(
        "--run", action="store_true", help="Run server only (skip checks)"
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="auto",
        help=f"CUDA version for PyTorch (auto, cpu, or {cuda_versions})",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    # Set interactive mode (default is interactive, --auto disables it)
    interactive = not args.auto

    # Print banner
    print_banner()

    # =========================================================================
    # Step 1: Python Version Check
    # =========================================================================
    print_header("Step 1: System Requirements")

    ok, msg = check_python_version()
    if ok:
        print_success(msg)
    else:
        print_fail(msg)
        print()
        print_fail("Please install Python 3.12 or later from https://python.org")
        return 1

    # =========================================================================
    # Step 2: Package Manager Detection
    # =========================================================================
    pkg_manager, pkg_path = find_package_manager()
    if pkg_manager == "uv":
        print_success(f"Package manager: uv ({pkg_path})")
    elif pkg_manager == "pip":
        print_success(f"Package manager: pip ({pkg_path})")
        print_info("Consider installing 'uv' for faster dependency installation")
    else:
        print_fail("No package manager found (uv or pip)")
        return 1

    # =========================================================================
    # Step 3: CUDA Detection
    # =========================================================================
    if args.cuda == "auto":
        detected_cuda = detect_cuda_version()
        if detected_cuda:
            cuda_version = detected_cuda
            print_success(f"CUDA detected: {cuda_version}")
        else:
            cuda_version = None
            print_warn("NVIDIA GPU not detected, using CPU-only PyTorch")
    elif args.cuda.lower() == "cpu":
        cuda_version = None
        print_info("Using CPU-only PyTorch (--cuda cpu)")
    else:
        cuda_version = args.cuda
        print_info(f"Using specified CUDA version: {cuda_version}")

    pytorch_index, pytorch_label = get_pytorch_index(cuda_version)
    print_info(f"PyTorch index: {pytorch_label}")

    # =========================================================================
    # Step 4: Virtual Environment
    # =========================================================================
    if args.run:
        # Skip setup, just run
        if not get_venv_python().exists():
            print_fail(
                "Virtual environment not found. Run 'python run.py --install' first."
            )
            return 1
    else:
        venv_exists = VENV_DIR.exists() and get_venv_python().exists()
        if venv_exists:
            print_header("Step 2: Virtual Environment")
            print_success(f"Virtual environment exists at {VENV_DIR}")
        else:
            if interactive and not prompt_yn("Create virtual environment?"):
                print_info("Skipping virtual environment creation")
                return 0
            if not create_venv(pkg_manager, step_num=2):
                return 1

    # =========================================================================
    # Step 3: Create Required Directories
    # =========================================================================
    if not args.run and not args.check:
        if not create_directories(step_num=3):
            return 1

    # =========================================================================
    # Step 4: Install Dependencies
    # =========================================================================
    if args.install or (not args.check and not args.run and not args.download):
        # Check if already installed
        venv_python = get_venv_python()
        code, stdout, _ = run_command(
            [
                str(venv_python),
                "-c",
                'import uvicorn; import torch; import transformers; print("OK")',
            ],
            capture=True,
        )
        already_installed = code == 0 and "OK" in stdout

        if already_installed:
            print_header("Step 4: Dependencies")
            print_success("Dependencies already installed")
        else:
            if interactive and not prompt_yn(
                f"Install dependencies ({pytorch_label})?"
            ):
                print_info("Skipping dependency installation")
            else:
                if not install_dependencies(
                    pkg_manager, pytorch_index, pytorch_label, step_num=4
                ):
                    return 1

    # =========================================================================
    # Step 5: Download TTS Models
    # =========================================================================
    if args.download or (not args.check and not args.run and not args.install):
        # Check if already downloaded (expect 87 Piper voices)
        voices_dir = SCRIPT_DIR / "tts_voices"
        piper_voices = list(voices_dir.glob("*.onnx")) if voices_dir.exists() else []
        already_downloaded = (
            len(piper_voices) >= 80
        )  # At least 80 of 87 expected voices

        if already_downloaded:
            print_header("Step 5: TTS Models")
            print_success(
                f"TTS voices already downloaded ({len(piper_voices)} Piper voices)"
            )
        else:
            if interactive and not prompt_yn("Download TTS models (~5GB)?"):
                print_info("Skipping TTS model download")
            else:
                if not download_tts_models(step_num=5):
                    print_warn("TTS model download had issues, but continuing...")

    # =========================================================================
    # Step 6: Verify Installation
    # =========================================================================
    if args.check or (not args.run):
        if interactive and not args.check:
            if not prompt_yn("Verify installation?"):
                print_info("Skipping verification")
            else:
                if not verify_installation(step_num=6):
                    print_warn("Some verification checks failed")
        else:
            if not verify_installation(step_num=6):
                if args.check:
                    return 1
                print_warn(
                    "Some verification checks failed, but attempting to continue..."
                )

    # =========================================================================
    # Step 7: Run Server
    # =========================================================================
    if args.check or args.install or args.download:
        print()
        print_header("Setup Complete!")
        print_success("WakeBuilder is ready to run")
        print_info("Start the server with: python run.py --run")
        print_info("Or: python run.py --auto")
        return 0

    if interactive and not prompt_yn("Start the server?"):
        print_info("Setup complete. Run 'python run.py --run' to start the server.")
        return 0

    print()
    print_header("Step 7: Starting Server")
    run_server(args.host, args.port)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print_info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_fail(f"Unexpected error: {e}")
        sys.exit(1)
