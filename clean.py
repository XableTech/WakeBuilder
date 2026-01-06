#!/usr/bin/env python3
"""
WakeBuilder Cleanup Script

A utility to clean up WakeBuilder data, caches, and downloaded models.
Use this to reset your environment or free up disk space.

Usage:
    python clean.py              # Interactive mode (prompts for each item)
    python clean.py --all        # Remove everything (use with caution!)
    python clean.py --venv       # Remove virtual environment only
    python clean.py --voices     # Remove Piper TTS voices only
    python clean.py --cache      # Remove HuggingFace cache only
    python clean.py --data       # Remove data directory only
    python clean.py --models     # Remove trained models only
    python clean.py --help       # Show help

This script is designed for:
- Resetting to a fresh state
- Freeing disk space
- Troubleshooting corrupted downloads
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional

# =============================================================================
# Constants
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
VENV_DIR = SCRIPT_DIR / ".venv"

# Directories that can be cleaned
CLEANABLE_DIRS = {
    "venv": {
        "path": VENV_DIR,
        "description": "Virtual environment (.venv)",
        "size_hint": "~2-5 GB",
    },
    "voices": {
        "path": SCRIPT_DIR / "tts_voices",
        "description": "Piper TTS voices",
        "size_hint": "~5 GB (87 voices)",
    },
    "data": {
        "path": SCRIPT_DIR / "data",
        "description": "Training data and negative samples",
        "size_hint": "~500 MB - 2 GB",
    },
    "models": {
        "path": SCRIPT_DIR / "models",
        "description": "Trained wake word models",
        "size_hint": "Varies",
    },
    "recordings": {
        "path": SCRIPT_DIR / "recordings",
        "description": "User recordings from web UI",
        "size_hint": "Varies",
    },
}

# External caches
CACHE_DIRS = {
    "huggingface": {
        "path": Path.home() / ".cache" / "huggingface",
        "description": "HuggingFace models (Kokoro, AST)",
        "size_hint": "~5-10 GB",
    },
    "coqui": {
        # Windows: %LOCALAPPDATA%\tts, Linux/Mac: ~/.local/share/tts
        "path": Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "tts",
        "description": "Coqui TTS models",
        "size_hint": "~2-5 GB",
    },
    "torch": {
        "path": Path.home() / ".cache" / "torch",
        "description": "PyTorch hub models",
        "size_hint": "~1-2 GB",
    },
    "piper": {
        "path": Path.home() / ".local" / "share" / "piper_tts",
        "description": "Piper TTS cache",
        "size_hint": "~100 MB",
    },
}

# =============================================================================
# Terminal Styling (copy from run.py)
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    ENABLED = (
        hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
        os.environ.get('TERM', '') != 'dumb' and
        os.environ.get('NO_COLOR') is None
    )
    
    RESET = '\033[0m' if ENABLED else ''
    BOLD = '\033[1m' if ENABLED else ''
    DIM = '\033[2m' if ENABLED else ''
    RED = '\033[91m' if ENABLED else ''
    GREEN = '\033[92m' if ENABLED else ''
    YELLOW = '\033[93m' if ENABLED else ''
    BLUE = '\033[94m' if ENABLED else ''
    CYAN = '\033[96m' if ENABLED else ''
    WHITE = '\033[97m' if ENABLED else ''


class Symbols:
    """Unicode symbols with ASCII fallbacks."""
    try:
        '✓✗→•'.encode(sys.stdout.encoding or 'utf-8')
        CHECK = '✓'
        CROSS = '✗'
        ARROW = '→'
        BULLET = '•'
        WARN = '⚠️'
        INFO = 'ℹ️'
        TRASH = '🗑️'
    except (UnicodeEncodeError, LookupError):
        CHECK = '[OK]'
        CROSS = '[X]'
        ARROW = '->'
        BULLET = '*'
        WARN = '[!]'
        INFO = '[i]'
        TRASH = '[DEL]'


def print_header(text: str) -> None:
    """Print a styled header."""
    width = 70
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print()


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


def print_item(text: str) -> None:
    """Print a list item."""
    print(f"  {Colors.WHITE}{Symbols.BULLET} {text}{Colors.RESET}")


def prompt_yn(question: str, default: bool = False) -> bool:
    """Prompt user for yes/no confirmation."""
    if default:
        prompt = f"  {Colors.YELLOW}? {question} [Y/n]: {Colors.RESET}"
    else:
        prompt = f"  {Colors.YELLOW}? {question} [y/N]: {Colors.RESET}"
    
    try:
        response = input(prompt).strip().lower()
        if not response:
            return default
        return response in ('y', 'yes', 'ye', '1', 'true')
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_banner() -> None:
    """Print the cleanup banner."""
    banner = f"""
{Colors.RED}{Colors.BOLD}
    ██████╗██╗     ███████╗ █████╗ ███╗   ██╗██╗   ██╗██████╗ 
   ██╔════╝██║     ██╔════╝██╔══██╗████╗  ██║██║   ██║██╔══██╗
   ██║     ██║     █████╗  ███████║██╔██╗ ██║██║   ██║██████╔╝
   ██║     ██║     ██╔══╝  ██╔══██║██║╚██╗██║██║   ██║██╔═══╝ 
   ╚██████╗███████╗███████╗██║  ██║██║ ╚████║╚██████╔╝██║     
    ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     
{Colors.RESET}
{Colors.DIM}    WakeBuilder Cleanup Utility{Colors.RESET}
    """
    print(banner)


# =============================================================================
# Utility Functions
# =============================================================================

def get_dir_size(path: Path) -> str:
    """Get human-readable directory size."""
    if not path.exists():
        return "0 B"
    
    try:
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        
        # Convert to human-readable
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} PB"
    except (PermissionError, OSError):
        return "? (access denied)"


def remove_directory(path: Path, description: str) -> bool:
    """Remove a directory and report status."""
    if not path.exists():
        print_info(f"{description} - not found (skipping)")
        return True
    
    size = get_dir_size(path)
    try:
        shutil.rmtree(path)
        print_success(f"{description} - removed ({size})")
        return True
    except PermissionError:
        print_fail(f"{description} - permission denied")
        return False
    except Exception as e:
        print_fail(f"{description} - error: {e}")
        return False


# =============================================================================
# Main Functions
# =============================================================================

def show_status() -> None:
    """Show current disk usage of cleanable directories."""
    print_header("Current Disk Usage")
    
    total_size = 0
    
    print(f"\n  {Colors.BOLD}Project Directories:{Colors.RESET}")
    for key, info in CLEANABLE_DIRS.items():
        path = info["path"]
        if path.exists():
            size = get_dir_size(path)
            # Parse size for total
            try:
                val, unit = size.split()
                multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
                total_size += float(val) * multipliers.get(unit, 1)
            except:
                pass
            print(f"    {Symbols.BULLET} {info['description']}: {Colors.CYAN}{size}{Colors.RESET}")
        else:
            print(f"    {Symbols.BULLET} {info['description']}: {Colors.DIM}(not created){Colors.RESET}")
    
    print(f"\n  {Colors.BOLD}External Caches:{Colors.RESET}")
    for key, info in CACHE_DIRS.items():
        path = info["path"]
        if path.exists():
            size = get_dir_size(path)
            try:
                val, unit = size.split()
                multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
                total_size += float(val) * multipliers.get(unit, 1)
            except:
                pass
            print(f"    {Symbols.BULLET} {info['description']}: {Colors.CYAN}{size}{Colors.RESET}")
        else:
            print(f"    {Symbols.BULLET} {info['description']}: {Colors.DIM}(not found){Colors.RESET}")
    
    # Format total
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_size < 1024:
            total_str = f"{total_size:.1f} {unit}"
            break
        total_size /= 1024
    else:
        total_str = f"{total_size:.1f} PB"
    
    print(f"\n  {Colors.BOLD}Total:{Colors.RESET} {Colors.YELLOW}{total_str}{Colors.RESET}")


def clean_interactive() -> int:
    """Interactive cleanup - prompt for each item."""
    print_header("Interactive Cleanup")
    
    removed = 0
    skipped = 0
    failed = 0
    
    # Project directories
    print(f"\n  {Colors.BOLD}Project Directories:{Colors.RESET}\n")
    for key, info in CLEANABLE_DIRS.items():
        path = info["path"]
        if not path.exists():
            continue
        
        size = get_dir_size(path)
        if prompt_yn(f"Remove {info['description']} ({size})?"):
            if remove_directory(path, info['description']):
                removed += 1
            else:
                failed += 1
        else:
            print_info(f"Skipping {info['description']}")
            skipped += 1
    
    # External caches
    print(f"\n  {Colors.BOLD}External Caches:{Colors.RESET}\n")
    for key, info in CACHE_DIRS.items():
        path = info["path"]
        if not path.exists():
            continue
        
        size = get_dir_size(path)
        if prompt_yn(f"Remove {info['description']} ({size})?"):
            if remove_directory(path, info['description']):
                removed += 1
            else:
                failed += 1
        else:
            print_info(f"Skipping {info['description']}")
            skipped += 1
    
    # Summary
    print_header("Cleanup Complete")
    print_success(f"Removed: {removed} items")
    print_info(f"Skipped: {skipped} items")
    if failed > 0:
        print_fail(f"Failed: {failed} items")
    
    return 0 if failed == 0 else 1


def clean_all() -> int:
    """Remove everything."""
    print_header("Full Cleanup")
    print_warn("This will remove ALL WakeBuilder data and caches!")
    print()
    
    if not prompt_yn("Are you sure you want to continue?"):
        print_info("Aborted.")
        return 0
    
    removed = 0
    failed = 0
    
    # Project directories
    for key, info in CLEANABLE_DIRS.items():
        if remove_directory(info["path"], info["description"]):
            removed += 1
        else:
            failed += 1
    
    # External caches
    for key, info in CACHE_DIRS.items():
        if remove_directory(info["path"], info["description"]):
            removed += 1
        else:
            failed += 1
    
    print_header("Cleanup Complete")
    print_success(f"Removed: {removed} items")
    if failed > 0:
        print_fail(f"Failed: {failed} items")
    
    return 0 if failed == 0 else 1


def clean_specific(targets: list[str]) -> int:
    """Remove specific items."""
    print_header("Selective Cleanup")
    
    removed = 0
    failed = 0
    
    for target in targets:
        if target in CLEANABLE_DIRS:
            info = CLEANABLE_DIRS[target]
            if remove_directory(info["path"], info["description"]):
                removed += 1
            else:
                failed += 1
        elif target in CACHE_DIRS:
            info = CACHE_DIRS[target]
            if remove_directory(info["path"], info["description"]):
                removed += 1
            else:
                failed += 1
        else:
            print_warn(f"Unknown target: {target}")
    
    print_header("Cleanup Complete")
    print_success(f"Removed: {removed} items")
    if failed > 0:
        print_fail(f"Failed: {failed} items")
    
    return 0 if failed == 0 else 1


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='WakeBuilder Cleanup Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean.py              Interactive mode (prompts for each item)
  python clean.py --all        Remove everything
  python clean.py --venv       Remove virtual environment
  python clean.py --voices     Remove Piper TTS voices
  python clean.py --cache      Remove HuggingFace cache
  python clean.py --status     Show disk usage only
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Remove all data and caches')
    parser.add_argument('--venv', action='store_true',
                        help='Remove virtual environment')
    parser.add_argument('--voices', action='store_true',
                        help='Remove Piper TTS voices')
    parser.add_argument('--data', action='store_true',
                        help='Remove training data')
    parser.add_argument('--models', action='store_true',
                        help='Remove trained models')
    parser.add_argument('--recordings', action='store_true',
                        help='Remove user recordings')
    parser.add_argument('--cache', action='store_true',
                        help='Remove HuggingFace and PyTorch caches')
    parser.add_argument('--status', action='store_true',
                        help='Show disk usage status only')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Status only
    if args.status:
        show_status()
        return 0
    
    # Full cleanup
    if args.all:
        return clean_all()
    
    # Specific targets
    targets = []
    if args.venv:
        targets.append('venv')
    if args.voices:
        targets.append('voices')
    if args.data:
        targets.append('data')
    if args.models:
        targets.append('models')
    if args.recordings:
        targets.append('recordings')
    if args.cache:
        targets.extend(['huggingface', 'coqui', 'torch', 'piper'])
    
    if targets:
        return clean_specific(targets)
    
    # Default: interactive mode
    show_status()
    print()
    return clean_interactive()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print_info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_fail(f"Unexpected error: {e}")
        sys.exit(1)
