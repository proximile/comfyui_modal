import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict
import os
import modal
import random
import string
import threading
import time

CUI_UNAME="webui"
allchars = string.ascii_letters + string.digits
CUI_PASS = ''.join(random.choice(allchars) for _ in range(8))
FORCE_REBUILD=False

# Define a persistent volume for caching Hugging Face downloads (to avoid repeated large downloads)
lora_vol = modal.Volume.from_name("lora-vol")

# Build the container image with required packages and ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git",force_build=FORCE_REBUILD)  # for comfy-cli to clone repos
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.4.1")
    .pip_install("huggingface_hub[hf_transfer]==0.30.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Install ComfyUI using comfy-cli (no dev flag needed)
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia --version 0.3.44")
    .run_commands("comfy --recent node install ComfyUI-Impact-Pack")
    .run_commands("comfy --recent node install https://github.com/WainWong/ComfyUI-Loop-image")
    # Install basic auth custom node
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && "
        "git clone https://github.com/fofr/comfyui-basic-auth"
    )
    # Set ComfyUI authentication credentials
    .env({
        "COMFYUI_USERNAME": CUI_UNAME,
        "COMFYUI_PASSWORD": CUI_PASS,
    })
)

app = modal.App(name="comfyui-srv", image=image)

# Define the ComfyUI service class with web UI and API endpoints
@app.cls(
    gpu="A100",
    volumes={"/data": lora_vol},
    scaledown_window=900,  # keep container alive for 15 minutes after use
    max_containers=1,
)
@modal.concurrent(max_inputs=20)  # allow a few concurrent API calls in one container
class ComfyService:
    port: int = 8000  # port where ComfyUI will run internally

    def __init__(self):
        self.symlink_thread = None
        self.stop_symlink_thread = False

    @modal.enter()
    def launch_comfy(self):
        """Launch ComfyUI server in background on container start."""
        print("🔧 Setting up ComfyUI data directories...")
        self.prep_cui_data()
        
        print("🚀 Starting ComfyUI server...")
        cmd = f"comfy launch --background -- --listen 0.0.0.0 --port {self.port} --verbose"
        subprocess.run(cmd, shell=True, check=True)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        print("✅ ComfyUI server launched successfully")
        
        # Start the periodic symlink checker
        self.start_periodic_lora_monitor()
        
        # Test server responsiveness
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=10)
            print("✅ ComfyUI server is responding")
        except Exception as e:
            print(f"⚠️ ComfyUI server may not be fully ready: {e}")

    def start_periodic_lora_monitor(self):
        """Start a background thread that monitors for new lora files every minute."""
        print("🔗 Starting periodic LoRA monitor...")
        self.stop_symlink_thread = False
        self.symlink_thread = threading.Thread(target=self._monitor_lora_files, daemon=True)
        self.symlink_thread.start()

    def _monitor_lora_files(self):
        """Background thread that monitors for new lora files and creates symlinks."""
        while not self.stop_symlink_thread:
            try:
                self.update_lora_symlinks()
            except Exception as e:
                print(f"⚠️ Error in LoRA monitor: {e}")
            
            # Wait 60 seconds before next check
            for _ in range(60):
                if self.stop_symlink_thread:
                    break
                time.sleep(1)

    def update_lora_symlinks(self):
        """Check for new safetensor files in runs folder and create symlinks."""
        from pathlib import Path
        import sys
        
        comfy_root = Path("/root/comfy/ComfyUI")
        lora_dir = comfy_root / "models" / "loras"
        lora_dir.mkdir(parents=True, exist_ok=True)

        runs_root = Path("/data/runs")
        if not runs_root.exists():
            return

        new_files_count = 0
        
        # Get all existing symlinks to track what we already have
        existing_symlinks = {f.name for f in lora_dir.iterdir() if f.is_symlink()}
        
        for checkpoint in runs_root.glob("*/checkpoints/*.safetensors"):
            # parent structure: /data/runs/<run_name>/checkpoints/<file>
            run_name = checkpoint.parents[1].name
            dest_name = f"{run_name}_{checkpoint.name}"
            dest = lora_dir / dest_name
            
            # Skip if symlink already exists and points to the right place
            if dest_name in existing_symlinks:
                try:
                    if dest.resolve() == checkpoint.resolve():
                        continue  # Symlink already exists and is correct
                except:
                    pass  # If we can't resolve, recreate the symlink
            
            try:
                # Remove existing symlink/file if it exists
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                
                # Create the symlink
                dest.symlink_to(checkpoint)
                new_files_count += 1
                
                print(f"🔗 Created LoRA symlink: {dest_name}")
                
            except Exception as e:
                print(f"⚠️ Failed to create symlink for {checkpoint}: {e}")

        if new_files_count > 0:
            print(f"✅ Created {new_files_count} new LoRA symlinks")
            
            # Trigger ComfyUI to refresh its model cache
            try:
                # Add ComfyUI folder_paths to Python path and refresh the loras list
                sys.path.insert(0, str(comfy_root))
                import folder_paths
                
                # Force refresh of the LoRA file list by calling get_filename_list
                # This will invalidate the cache since we changed the directory
                folder_paths.get_filename_list("loras")
                print("🔄 Refreshed ComfyUI LoRA cache")
                
            except Exception as e:
                print(f"⚠️ Could not refresh ComfyUI cache: {e}")
    
    def prep_cui_data(self):
        """
        Prepare the ComfyUI data layout inside the container.

        - Replaces /root/comfy/ComfyUI/output with a symlink to /data/sample_outputs
        - Symlinks *.safetensors from /data/models → checkpoints/
        - Sets up initial LoRA symlinks (ongoing monitoring handled separately)
        """
        from pathlib import Path
        import shutil

        comfy_root = Path("/root/comfy/ComfyUI")
        output_dir = comfy_root / "output"
        sample_outputs = Path("/data/sample_outputs")

        # ---------- 1. output → sample_outputs ----------
        if output_dir.exists() or output_dir.is_symlink():
            # remove file, dir, or symlink uniformly
            if output_dir.is_dir() and not output_dir.is_symlink():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()

        # Create the symlink (ensure target exists to avoid broken link)
        sample_outputs.mkdir(parents=True, exist_ok=True)
        output_dir.symlink_to(sample_outputs, target_is_directory=True)

        # ---------- 2. model checkpoints ----------
        src_models_dir = Path("/data/models")
        dst_ckpts_dir = comfy_root / "models" / "checkpoints"
        dst_ckpts_dir.mkdir(parents=True, exist_ok=True)

        if src_models_dir.exists():
            for model_path in src_models_dir.glob("*.safetensors"):
                dest = dst_ckpts_dir / model_path.name
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(model_path)

        # ---------- 3. initial LoRA setup ----------
        print("📁 ComfyUI data setup complete (LoRA monitoring will start after server launch)")

    def poll_server_health(self):
        """Check if ComfyUI server is responding, otherwise stop container."""
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/system_stats", timeout=5)
            # If the request succeeds quickly, the server is healthy
        except Exception as e:
            print(f"Health check failed: {e}")
            # Stop fetching new inputs, container will shut down
            modal.experimental.stop_fetching_inputs()

    @modal.method()
    def infer(self, prompt: str) -> bytes:
        """Run a simple text-to-image generation and return image bytes."""
        # Ensure server is up
        self.poll_server_health()
        
        # Create a unique prefix for output file
        unique_id = uuid.uuid4().hex[:8]
        
        # Use comfy command line to run a simple prompt
        cmd = f'comfy run --prompt "{prompt}" --output-prefix {unique_id} --wait --timeout 1200'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ComfyUI generation failed: {result.stderr}")
            raise RuntimeError(f"ComfyUI generation failed: {result.stderr}")
        
        # Read the resulting image file
        output_dir = Path("/root/comfy/ComfyUI/output")
        for f in output_dir.iterdir():
            if f.name.startswith(unique_id) and f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                img_bytes = f.read_bytes()
                # Clean up the image file
                f.unlink(missing_ok=True)
                return img_bytes
        
        raise RuntimeError("Output image not found.")

    @modal.fastapi_endpoint(method="GET", label="status")
    def get_status(self):
        """Get server status and health information."""
        import urllib.request
        import json
        
        try:
            # Check if ComfyUI is responding
            response = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/system_stats", timeout=5)
            system_stats = json.loads(response.read().decode())
            
            # Count current LoRA symlinks
            lora_dir = Path("/root/comfy/ComfyUI/models/loras")
            lora_count = len([f for f in lora_dir.iterdir() if f.is_symlink()]) if lora_dir.exists() else 0
            
            return {
                "status": "healthy",
                "comfyui_port": self.port,
                "system_stats": system_stats,
                "lora_symlinks": lora_count,
                "monitor_active": self.symlink_thread is not None and self.symlink_thread.is_alive(),
                "endpoints": {
                    "generate": "/generate",
                    "status": "/status",
                    "ui": "/"
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "comfyui_port": self.port
            }

    @modal.fastapi_endpoint(method="POST", label="generate")
    def generate(self, request: Dict):
        """FastAPI endpoint to generate an image from a prompt. Expects JSON: {"prompt": "..."}"""
        prompt_text = request.get("prompt", "")
        if not prompt_text:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        try:
            print(f"🎨 Processing request: {prompt_text[:50]}...")
            image_bytes = self.infer.local(prompt_text)
            print(f"✅ Generated image ({len(image_bytes)} bytes)")
            from fastapi import Response
            # Return image as PNG
            return Response(image_bytes, media_type="image/png")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @modal.web_server(port, label="comfyui")
    def serve_ui(self):
        """Expose the ComfyUI web UI. Navigate to this endpoint in a browser."""
        import time
        
        # Wait a bit for ComfyUI to be ready
        time.sleep(2)
        
        # Check if ComfyUI is running
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5)
            return {"status": "ComfyUI UI server is running", "port": self.port}
        except Exception as e:
            return {"status": "ComfyUI UI server is starting", "error": str(e)}

@app.local_entrypoint()
def main():
    from modal import enable_output

    # First, stop any existing deployment to avoid conflicts
    print("🛑 Stopping any existing deployment...")
    try:
        app.stop()
    except Exception as e:
        print(f"No existing deployment to stop: {e}")
    
    # build + deploy (prints logs while building)
    with enable_output():
        app.deploy(name="comfyui_sdxl_app")

    # Fix deprecation warnings by properly instantiating the class
    service = ComfyService()
    ui_url = service.serve_ui.get_web_url()
    api_url = service.generate.get_web_url()
    status_url = service.get_status.get_web_url()

    print("=" * 60)
    print("🚀 ComfyUI SDXL Service Deployed Successfully!")
    print("📦 Models: SDXL Base 1.0 (development ready)")
    print("🔗 Auto LoRA symlink monitor: Enabled (every 60 seconds)")
    print("=" * 60)
    print(f"📱 UI     → {ui_url}")
    print(f"🔗 API    → {api_url}")
    print(f"📊 Status → {status_url}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Check server status (includes LoRA symlink count):")
    print(f"   curl {status_url}")
    print("2. Visit the UI URL in your browser")
    print("   - Use ComfyUI credentials: "+CUI_UNAME+" / "+CUI_PASS)
    print("3. Use the API URL directly (no auth required):")
    print(f"   curl -X POST '{api_url}' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"prompt\": \"A beautiful sunset over mountains\"}' \\")
    print("        --output generated_image.png")
    print("4. LoRA files automatically symlinked from /data/runs/*/checkpoints/")
    print("5. ComfyUI cache auto-refreshed when new LoRAs detected")
    print("=" * 60)

if __name__ == "__main__":
    main()
