import subprocess
import sys
import platform
import argparse
import os

def install_pytorch(use_cpu=False):
    """Install PyTorch based on platform and user preference."""
    system = platform.system()
    
    if use_cpu:
        print("Installing PyTorch (CPU-only version)...")
        subprocess.check_call([
            "uv", "pip", "install",
            "torch>=2.4.0", "torchvision>=0.19.0", "torchaudio>=2.4.0"
        ])
        return
    
    print(f"Detected platform: {system}")
    
    if system == "Darwin":  # macOS
        print("Installing PyTorch for macOS (MPS support)...")
        subprocess.check_call([
            "uv", "pip", "install",
            "torch>=2.4.0", "torchvision>=0.25.0", "torchaudio>=2.4.0"
        ])
    elif system in ["Windows", "Linux"]:
        print("Installing PyTorch with CUDA 12.4 support...")
        subprocess.check_call([
            "uv", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ])
    else:
        print(f"Unknown platform: {system}. Installing CPU version...")
        subprocess.check_call([
            "uv", "pip", "install",
            "torch>=2.4.0", "torchvision>=0.19.0", "torchaudio>=2.4.0"
        ])

def install_dependencies():
    """Install project dependencies."""
    print("Installing project dependencies...")
    subprocess.check_call(["uv", "pip", "install", "-e", "."])

def download_model():
    """Download the Dolphin-1.5 model from Hugging Face."""
    model_name = "ByteDance/Dolphin-1.5"
    model_dir = "hf_model"
    
    print(f"\nDownloading model {model_name} to {model_dir}...")
    
    # Check if model directory already exists
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Model directory '{model_dir}' already exists and is not empty.")
        response = input("Do you want to re-download? (y/n): ").lower()
        if response != 'y':
            print("Skipping model download.")
            return
    
    try:
        subprocess.check_call([
            "huggingface-cli", "download",
            model_name,
            "--local-dir", model_dir,
            "--local-dir-use-symlinks", "False"
        ])
        print(f"✓ Model downloaded successfully to {model_dir}")
    except FileNotFoundError:
        print("\n✗ Error: 'huggingface-cli' not found.")
        print("Installing huggingface-hub with CLI support...")
        subprocess.check_call(["uv", "pip", "install", "huggingface-hub[cli]"])
        # Retry download
        subprocess.check_call([
            "huggingface-cli", "download",
            model_name,
            "--local-dir", model_dir,
            "--local-dir-use-symlinks", "False"
        ])
        print(f"✓ Model downloaded successfully to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install Dolphin OCR with dependencies")
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only version of PyTorch")
    parser.add_argument("--skip-model", action="store_true", help="Skip downloading the model")
    args = parser.parse_args()
    
    try:
        install_pytorch(use_cpu=args.cpu)
        install_dependencies()
        
        if not args.skip_model:
            download_model()
        else:
            print("\nSkipping model download (--skip-model flag used)")
        
        print("\n✓ Installation complete!")
        if not args.cpu and platform.system() in ["Windows", "Linux"]:
            print("  Installed with CUDA 12.4 support")
        elif args.cpu:
            print("  Installed with CPU-only support")
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n✗ Error: 'uv' command not found. Please install uv first:")
        print("  pip install uv")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)