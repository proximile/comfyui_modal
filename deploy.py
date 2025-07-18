import modal
import subprocess
import os
from pathlib import Path
import time

# Configuration
USERNAME = "webui"
allchars = string.ascii_letters + string.digits
CUI_PASS = ''.join(random.choice(allchars) for _ in range(8))

# Create Modal app
app = modal.App("comfyui-web-interface")

# Define volume - only use existing lora-vol
lora_volume = modal.Volume.from_name("lora-vol", create_if_missing=False)

# Build the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "gcc",
        "g++",
        "ffmpeg",
    )
    .pip_install(
        "comfy-cli==1.4.1",
    )
    .run_commands(
        # Install ComfyUI
        "comfy --skip-prompt install --nvidia --version 0.3.44",
        # Install basic auth custom node
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/fofr/comfyui-basic-auth || true",
        # Create necessary directories
        "mkdir -p /root/comfy/ComfyUI/models/checkpoints",
        "mkdir -p /root/comfy/ComfyUI/models/loras",
        "mkdir -p /root/comfy/ComfyUI/output",
    )
    .run_commands("comfy --recent node install ComfyUI-Impact-Pack")
)

# Helper to set up model directories
def setup_model_dirs():
    """Set up model directories with proper symlinks."""
    import shutil
    
    comfyui_root = Path("/root/comfy/ComfyUI")
    models_dir = comfyui_root / "models"
    checkpoints_dir = models_dir / "checkpoints"
    loras_dir = models_dir / "loras"
    output_dir = comfyui_root / "output"
    
    # Ensure directories exist
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    loras_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove and recreate output directory as symlink to /data/sample_outputs
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    
    sample_outputs_dir = Path("/data/sample_outputs")
    sample_outputs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.symlink_to(sample_outputs_dir)
    
    print("📁 Setting up model symlinks...")
    
    # Symlink models from /data/models/*.safetensors to checkpoints
    data_models_dir = Path("/data/models")
    if data_models_dir.exists():
        model_count = 0
        for model_file in data_models_dir.glob("*.safetensors"):
            if model_file.is_file():
                symlink_path = checkpoints_dir / model_file.name
                if symlink_path.exists() or symlink_path.is_symlink():
                    symlink_path.unlink()
                symlink_path.symlink_to(model_file)
                print(f"  ✓ Linked checkpoint: {model_file.name}")
                model_count += 1
        print(f"✅ Created {model_count} checkpoint symlinks")
    
    # Symlink LoRAs from /data/runs/*/checkpoints/*.safetensors to loras
    runs_dir = Path("/data/runs")
    if runs_dir.exists():
        lora_count = 0
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                checkpoints_subdir = run_dir / "checkpoints"
                if checkpoints_subdir.exists():
                    for lora_file in checkpoints_subdir.glob("*.safetensors"):
                        symlink_name = f"{run_dir.name}_{lora_file.name}"
                        symlink_path = loras_dir / symlink_name
                        if symlink_path.exists() or symlink_path.is_symlink():
                            symlink_path.unlink()
                        symlink_path.symlink_to(lora_file)
                        lora_count += 1
                        if lora_count <= 10:  # Only print first 10
                            print(f"  ✓ Linked LoRA: {symlink_name}")
        if lora_count > 10:
            print(f"  ... and {lora_count - 10} more")
        print(f"✅ Created {lora_count} LoRA symlinks")

# ComfyUI service with just web interface
@app.cls(
    image=image,
    gpu="L4",
    volumes={
        "/data": lora_volume,
    },
    timeout=3600,
    max_containers=1,
    scaledown_window=900,  # Keep alive for 15 minutes
)
class ComfyUIWeb:
    
    @modal.enter()
    def startup(self):
        """Start ComfyUI on container startup."""
        print("🚀 Starting ComfyUI Web Interface...")
        
        # Set up model directories
        setup_model_dirs()
        
        # Set auth environment variables
        os.environ["COMFYUI_USERNAME"] = USERNAME
        os.environ["COMFYUI_PASSWORD"] = PASSWORD
        
        # Start ComfyUI in background
        cmd = [
            "python", "/root/comfy/ComfyUI/main.py",
            "--listen", "0.0.0.0",
            "--port", "8188",
            "--disable-auto-launch",
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        # Wait for ComfyUI to start
        print("⏳ Waiting for ComfyUI to initialize...")
        time.sleep(10)  # Give it 10 seconds to start
        
        # Check if process is still running
        if self.process.poll() is not None:
            # Process died, show output
            output, _ = self.process.communicate()
            print(f"❌ ComfyUI failed to start. Output:\n{output}")
            raise RuntimeError("ComfyUI process died during startup")
        else:
            print("✅ ComfyUI appears to be running!")
    
    @modal.exit()
    def shutdown(self):
        """Clean shutdown."""
        if hasattr(self, 'process') and self.process:
            self.process.terminate()
            self.process.wait()
    
    @modal.web_server(8188, startup_timeout=120)
    def serve(self):
        """Direct web server - just proxy the ComfyUI port."""
        # This is a pass-through - Modal will handle the proxying
        print("🌐 Web server endpoint active")

# Local entrypoint
@app.local_entrypoint()
def main():
    """Deploy the application."""
    print("🚀 Deploying ComfyUI Web Interface...")
    
    with modal.enable_output():
        app.deploy()
    
    # Get the URL
    service = ComfyUIWeb()
    web_url = service.serve.get_web_url()
    
    print("\n" + "="*70)
    print("✅ ComfyUI Web Interface deployed successfully!")
    print("="*70)
    print(f"🌐 Access URL: {web_url}")
    print(f"👤 Username: {USERNAME}")
    print(f"🔑 Password: {PASSWORD}")
    print("="*70)
    print("\n📝 Notes:")
    print("- The interface may take 30-60 seconds to fully load")
    print("- Basic auth is handled by the ComfyUI custom node")
    print("- Models are symlinked from /data/models/")
    print("- LoRAs are symlinked from /data/runs/*/checkpoints/")
    print("- Outputs are saved to /data/sample_outputs/")
    print("="*70)

if __name__ == "__main__":
    main()
