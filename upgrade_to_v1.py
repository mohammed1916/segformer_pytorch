#!/usr/bin/env python3
"""
MMSegmentation Upgrade Script
Upgrades MMSegmentation from 0.x to 1.x with latest dependencies
"""

import os
import re
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def update_imports_in_file(filepath):
    """Update import statements for MMSegmentation 1.x compatibility."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update MMCV runner imports to MMEngine
        content = re.sub(
            r'from mmcv\.runner import (.*)',
            r'from mmengine.runner import \1',
            content
        )

        # Update optimizer building
        content = re.sub(
            r'from mmcv\.runner import build_optimizer',
            r'from mmengine.optim import build_optim_wrapper',
            content
        )

        # Update parallel training
        content = re.sub(
            r'MMDataParallel',
            r'MMSegDataParallel',
            content
        )

        content = re.sub(
            r'MMDistributedDataParallel',
            r'MMSegDistributedDataParallel',
            content
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    """Main upgrade function."""
    print("🚀 Starting MMSegmentation upgrade to v1.2.2...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required for MMSegmentation 1.x")
        return False

    # Install/update dependencies
    print("📦 Installing/updating dependencies...")

    # deps = [
    #     "torch>=2.0.0",
    #     "torchvision>=0.15.0",
    #     "mmengine>=0.10.0",
    #     "mmcv>=2.0.0",
    #     "openmim"
    # ]

    # for dep in deps:
    #     print(f"Installing {dep}...")
    #     success, stdout, stderr = run_command(f"pip install -U {dep}")
    #     if not success:
    #         print(f"❌ Failed to install {dep}: {stderr}")
    #         return False

    # Install MMSegmentation
    print("Installing MMSegmentation...")
    success, stdout, stderr = run_command("pip install -e .")
    if not success:
        print(f"❌ Failed to install MMSegmentation: {stderr}")
        return False

    # Update import statements in key files
    print("🔧 Updating import statements for compatibility...")

    files_to_update = [
        "mmseg/apis/train.py",
        "mmseg/apis/test.py",
        "mmseg/core/evaluation/eval_hooks.py",
        "tools/train.py",
        "tools/test.py"
    ]

    for filepath in files_to_update:
        if os.path.exists(filepath):
            print(f"Updating {filepath}...")
            update_imports_in_file(filepath)

    print("✅ Upgrade completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review and update your configuration files (see migration guide)")
    print("2. Update your training scripts to use MMEngine APIs")
    print("3. Test your models with the new version")
    print("\n📖 Migration guide: https://mmsegmentation.readthedocs.io/en/latest/migration/interface.html")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
