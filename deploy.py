import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict
import os
import modal
import random
import string

CUI_UNAME="webui"
allchars = string.ascii_letters + string.digits
CUI_PASS = ''.join(random.choice(allchars) for _ in range(8))
FORCE_REBUILD=False

# Define a persistent volume for caching Hugging Face downloads (to avoid repeated large downloads)
lora_vol = modal.Volume.from_name("lora-vol")

# Use base64 encoding to avoid Docker parsing issues with Python imports
import base64

LORA_MONITOR_SCRIPT = '''#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from datetime import datetime

LORA_DIR = Path("/root/comfy/ComfyUI/models/loras")
RUNS_DIR = Path("/data/runs")
LOG_FILE = Path("/tmp/lora_symlinks.log")

def log_msg(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\\n")

def main():
    # Create lora directory if it doesn't exist
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if runs directory exists
    if not RUNS_DIR.exists():
        sys.exit(0)
    
    new_links = 0
    
    # Find all .safetensors files in runs/*/checkpoints/ directories
    for safetensor_file in RUNS_DIR.glob("*/checkpoints/*.safetensors"):
        # Extract run name from path
        run_name = safetensor_file.parents[1].name
        filename = safetensor_file.name
        
        # Create symlink name: run_name_filename
        symlink_name = f"{run_name}_{filename}"
        symlink_path = LORA_DIR / symlink_name
        
        # Check if symlink already exists and points to the correct file
        if symlink_path.is_symlink():
            try:
                if symlink_path.resolve() == safetensor_file.resolve():
                    continue  # Symlink already exists and is correct
            except:
                pass  # If resolve fails, recreate the symlink
        
        # Remove existing symlink/file if it exists
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        
        # Create the symlink
        try:
            symlink_path.symlink_to(safetensor_file)
            log_msg(f"Created symlink: {symlink_name} -> {safetensor_file}")
            new_links += 1
        except Exception as e:
            log_msg(f"Failed to create symlink: {symlink_name} - {e}")
    
    # Log summary if new links were created
    if new_links > 0:
        log_msg(f"Created {new_links} new LoRA symlinks")

if __name__ == "__main__":
    main()
'''

# Encode the script to base64 to avoid Docker parsing issues
LORA_SCRIPT_B64 = base64.b64encode(LORA_MONITOR_SCRIPT.encode('utf-8')).decode('ascii')

# Build the container image with required packages and ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cron", force_build=FORCE_REBUILD)  # Added cron
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
    # Create directory and write the Python script using base64 to avoid parsing issues
    .run_commands("mkdir -p /usr/local/bin")
    .run_commands(f'echo "{LORA_SCRIPT_B64}" | base64 -d > /usr/local/bin/update_lora_symlinks.py')
    .run_commands("chmod +x /usr/local/bin/update_lora_symlinks.py")
)

app = modal.App(name="comfyui-srv", image=image)

# Define the ComfyUI service class with web UI and API endpoints
@app.cls(
    gpu="L4",
    volumes={"/data": lora_vol},
    scaledown_window=900,  # keep container alive for 15 minutes after use
    max_containers=1,
)
@modal.concurrent(max_inputs=20)  # allow a few concurrent API calls in one container
class ComfyService:
    port: int = 8000  # port where ComfyUI will run internally

    @modal.enter()
    def launch_comfy(self):
        """Launch ComfyUI server in background on container start."""
        print("🔧 Setting up ComfyUI data directories...")
        self.prep_cui_data()
        
        print("⏰ Setting up cron job for LoRA monitoring (Python script)...")
        self.setup_lora_cron()
        
        print("🚀 Starting ComfyUI server...")
        cmd = f"comfy launch --background -- --listen 0.0.0.0 --port {self.port} --verbose"
        subprocess.run(cmd, shell=True, check=True)
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        print("✅ ComfyUI server launched successfully")
        
        # Test server responsiveness
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=10)
            print("✅ ComfyUI server is responding")
        except Exception as e:
            print(f"⚠️ ComfyUI server may not be fully ready: {e}")

    def setup_lora_cron(self):
        """Set up a cron job to check for new LoRA files every minute."""
        try:
            # Find the correct Python executable path
            python_path = self.find_python_executable()
            
            # Start the cron daemon
            subprocess.run(["service", "cron", "start"], check=False)
            
            # Add cron job to run every minute using the correct Python path
            cron_entry = f"* * * * * {python_path} /usr/local/bin/update_lora_symlinks.py\n"
            
            # Write to crontab
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(input=cron_entry)
            
            # Run the script once immediately to catch any existing files
            subprocess.run([python_path, "/usr/local/bin/update_lora_symlinks.py"], check=False)
            
            print("✅ LoRA monitoring cron job configured (runs Python script every minute)")
            print(f"✅ Using Python at: {python_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to set up cron job: {e}")
    
    def find_python_executable(self):
        """Find the correct Python executable path in the container."""
        # Try common Python paths in order of preference
        python_paths = [
            "/usr/local/bin/python3",
            "/usr/bin/python3", 
            "/opt/miniconda3/bin/python3",
            "/usr/local/bin/python",
            "/usr/bin/python"
        ]
        
        for path in python_paths:
            if Path(path).exists():
                return path
        
        # If none of the standard paths work, use 'which' to find python3
        try:
            result = subprocess.run(["which", "python3"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
            
        # Last resort: try using python3 from PATH
        return "python3"
    
    def prep_cui_data(self):
        """
        Prepare the ComfyUI data layout inside the container.

        - Replaces /root/comfy/ComfyUI/output with a symlink to /data/sample_outputs
        - Symlinks *.safetensors from /data/models → checkpoints/
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

        print("📁 ComfyUI data setup complete")

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
            
            # Check cron status
            cron_status = "unknown"
            try:
                result = subprocess.run(["service", "cron", "status"], capture_output=True, text=True)
                cron_status = "running" if result.returncode == 0 else "stopped"
            except:
                pass
            
            # Get last few log entries if available
            log_entries = []
            try:
                with open("/tmp/lora_symlinks.log", "r") as f:
                    log_entries = f.readlines()[-5:]  # Last 5 entries
            except:
                pass
            
            return {
                "status": "healthy",
                "comfyui_port": self.port,
                "system_stats": system_stats,
                "lora_symlinks": lora_count,
                "cron_status": cron_status,
                "recent_logs": [line.strip() for line in log_entries],
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

    @modal.fastapi_endpoint(method="GET", label="lora-logs")
    def get_lora_logs(self):
        """Get the full LoRA symlink logs."""
        try:
            with open("/tmp/lora_symlinks.log", "r") as f:
                logs = f.read()
            return {"logs": logs}
        except FileNotFoundError:
            return {"logs": "No logs found yet"}
        except Exception as e:
            return {"error": str(e)}

    @modal.fastapi_endpoint(method="POST", label="trigger-lora-scan")
    def trigger_lora_scan(self):
        """Manually trigger a LoRA scan."""
        try:
            python_path = self.find_python_executable()
            result = subprocess.run([python_path, "/usr/local/bin/update_lora_symlinks.py"], 
                                  capture_output=True, text=True, timeout=30)
            
            # Count current LoRA symlinks
            lora_dir = Path("/root/comfy/ComfyUI/models/loras")
            lora_count = len([f for f in lora_dir.iterdir() if f.is_symlink()]) if lora_dir.exists() else 0
            
            return {
                "success": True,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "lora_count": lora_count,
                "python_path": python_path
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Scan timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
    logs_url = service.get_lora_logs.get_web_url()
    trigger_url = service.trigger_lora_scan.get_web_url()

    print("=" * 60)
    print("🚀 ComfyUI SDXL Service Deployed Successfully!")
    print("📦 Models: SDXL Base 1.0 (development ready)")
    print("⏰ LoRA Monitor: Cron job (Python script every minute)")
    print("=" * 60)
    print(f"📱 UI        → {ui_url}")
    print(f"🔗 API       → {api_url}")
    print(f"📊 Status    → {status_url}")
    print(f"📋 LoRA Logs → {logs_url}")
    print(f"🔄 Trigger   → {trigger_url}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Check server status (includes LoRA symlink count & cron status):")
    print(f"   curl {status_url}")
    print("2. View LoRA monitor logs:")
    print(f"   curl {logs_url}")
    print("3. Manually trigger LoRA scan:")
    print(f"   curl -X POST {trigger_url}")
    print("4. Visit the UI URL in your browser")
    print("   - Use ComfyUI credentials: "+CUI_UNAME+" / "+CUI_PASS)
    print("5. Use the API URL directly (no auth required):")
    print(f"   curl -X POST '{api_url}' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"prompt\": \"A beautiful sunset over mountains\"}' \\")
    print("        --output generated_image.png")
    print("6. LoRA files automatically symlinked from /data/runs/*/checkpoints/")
    print("   - Python cron job runs every minute")
    print("   - Logs available at /tmp/lora_symlinks.log")
    print("=" * 60)

if __name__ == "__main__":
    main()
