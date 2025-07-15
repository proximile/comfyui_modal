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

    @modal.enter()
    def launch_comfy(self):
        """Launch ComfyUI server in background on container start."""
        print("🔧 Setting up ComfyUI data directories...")
        self.prep_cui_data()
        
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
    
    
    def prep_cui_data(self):
        """
        Prepare the ComfyUI data layout inside the container.

        - Replaces /root/comfy/ComfyUI/output with a symlink to /data/sample_outputs
        - Symlinks *.safetensors from /data/models → checkpoints/
        - Symlinks *.safetensors from /data/runs/<run>/checkpoints → loras/,
          naming them <run>_<original_file>
        """
        from pathlib import Path
        import shutil
        import os

        comfy_root = Path("/root/comfy/ComfyUI")
        output_dir = comfy_root / "output"
        sample_outputs = Path("/data/sample_outputs")

        # ---------- 1 – 2.  output → sample_outputs ----------
        if output_dir.exists() or output_dir.is_symlink():
            # remove file, dir, or symlink uniformly
            if output_dir.is_dir() and not output_dir.is_symlink():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()

        # Create the symlink (ensure target exists to avoid broken link)
        sample_outputs.mkdir(parents=True, exist_ok=True)
        output_dir.symlink_to(sample_outputs, target_is_directory=True)

        # ---------- 3. model checkpoints ----------
        src_models_dir = Path("/data/models")
        dst_ckpts_dir = comfy_root / "models" / "checkpoints"
        dst_ckpts_dir.mkdir(parents=True, exist_ok=True)

        if src_models_dir.exists():
            for model_path in src_models_dir.glob("*.safetensors"):
                dest = dst_ckpts_dir / model_path.name
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(model_path)

        # ---------- 4. lora checkpoints ----------
        lora_dir = comfy_root / "models" / "loras"
        lora_dir.mkdir(parents=True, exist_ok=True)

        runs_root = Path("/data/runs")
        if runs_root.exists():
            for checkpoint in runs_root.glob("*/checkpoints/*.safetensors"):
                # parent structure: /data/runs/<run_name>/checkpoints/<file>
                run_name = checkpoint.parents[1].name
                dest_name = f"{run_name}_{checkpoint.name}"
                dest = lora_dir / dest_name
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(checkpoint)

    def poll_server_health(self):
        """Check if ComfyUI server is responding, otherwise stop container."""
        import socket, urllib.request, urllib.error
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/system_stats", timeout=5)
            # If the request succeeds quickly, the server is healthy
        except Exception as e:
            print(f"Health check failed: {e}")
            # Stop fetching new inputs, container will shut down
            modal.experimental.stop_fetching_inputs()

    @modal.method()
    def infer(self, prompt: str) -> bytes:
        """Run the ComfyUI workflow for the given prompt and return image bytes."""
        # Ensure server is up
        self.poll_server_health()
        
        # Create a basic workflow programmatically if workflow_api.json doesn't exist
        workflow_path = "/root/workflow_api.json"
        if not Path(workflow_path).exists():
            # Create a simple SDXL workflow
            workflow_data = {
                "1": {
                    "inputs": {
                        "text": prompt,
                        "clip": ["4", 1]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Prompt)"}
                },
                "2": {
                    "inputs": {
                        "text": "blurry, low quality, distorted",
                        "clip": ["4", 1]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Negative)"}
                },
                "3": {
                    "inputs": {
                        "seed": 42,
                        "steps": 20,
                        "cfg": 7.0,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                        "model": ["4", 0],
                        "positive": ["1", 0],
                        "negative": ["2", 0],
                        "latent_image": ["5", 0]
                    },
                    "class_type": "KSampler",
                    "_meta": {"title": "KSampler"}
                },
                "4": {
                    "inputs": {
                        "ckpt_name": "sd_xl_base_1.0.safetensors"
                    },
                    "class_type": "CheckpointLoaderSimple",
                    "_meta": {"title": "Load Checkpoint"}
                },
                "5": {
                    "inputs": {
                        "width": 1024,
                        "height": 1024,
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage",
                    "_meta": {"title": "Empty Latent Image"}
                },
                "6": {
                    "inputs": {
                        "samples": ["3", 0],
                        "vae": ["4", 2]
                    },
                    "class_type": "VAEDecode",
                    "_meta": {"title": "VAE Decode"}
                },
                "7": {
                    "inputs": {
                        "filename_prefix": "generated_image",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage",
                    "_meta": {"title": "Save Image"}
                }
            }
            Path(workflow_path).write_text(json.dumps(workflow_data, indent=2))
        else:
            # Load existing workflow
            workflow_data = json.loads(Path(workflow_path).read_text())
        
        # Insert prompt text into the ClipTextEncode node (id "1")
        workflow_data["1"]["inputs"]["text"] = prompt
        # Create a unique prefix for output file
        unique_id = uuid.uuid4().hex
        # Find the SaveImage node and update filename_prefix
        for node_id, node_data in workflow_data.items():
            if node_data.get("class_type") == "SaveImage":
                node_data["inputs"]["filename_prefix"] = unique_id
                break
        else:
            # If no SaveImage node found, update node "7" (our default)
            if "7" in workflow_data:
                workflow_data["7"]["inputs"]["filename_prefix"] = unique_id
        
        # Save this modified workflow to a temp file
        temp_path = f"/root/{unique_id}.json"
        Path(temp_path).write_text(json.dumps(workflow_data))
        
        # Run the workflow via comfy CLI
        run_cmd = f"comfy run --workflow {temp_path} --wait --timeout 1200"
        result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ComfyUI workflow failed: {result.stderr}")
            raise RuntimeError(f"ComfyUI workflow failed: {result.stderr}")
        
        # Read the resulting image file
        output_dir = Path("/root/comfy/ComfyUI/output")
        for f in output_dir.iterdir():
            if f.name.startswith(unique_id) and f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                img_bytes = f.read_bytes()
                # Clean up temp files
                Path(temp_path).unlink(missing_ok=True)
                f.unlink(missing_ok=True)
                return img_bytes
        
        # Clean up temp file even if no output found
        Path(temp_path).unlink(missing_ok=True)
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
            
            return {
                "status": "healthy",
                "comfyui_port": self.port,
                "system_stats": system_stats,
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
        """
        Expose the ComfyUI web UI. Navigate to this endpoint in a browser.
        """
        import subprocess
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

# Test function that doesn't require proxy auth
@app.function()
def test_endpoints(ui_url: str, api_url: str, status_url: str):
    """Test function that tests the publicly accessible endpoints."""
    import time, requests
    
    print(f"ComfyUI web UI available at: {ui_url}")
    print(f"ComfyUI API available at: {api_url}")
    print(f"ComfyUI Status available at: {status_url}")
    
    # Test status endpoint first
    print("Testing status endpoint...")
    try:
        res = requests.get(status_url, timeout=10)
        if res.status_code == 200:
            status_data = res.json()
            print(f"✅ Status: {status_data.get('status', 'unknown')}")
        else:
            print(f"⚠️ Status endpoint returned {res.status_code}")
    except Exception as e:
        print(f"❌ Status endpoint failed: {e}")
    
    # Wait for ComfyUI server to be fully ready
    print("Waiting for ComfyUI server to be ready...")
    for i in range(30):
        try:
            res = requests.get(f"{ui_url}/", timeout=10)
            if res.status_code == 200:
                print("ComfyUI UI endpoint is responding")
                break
        except Exception as e:
            print(f"Attempt {i+1}/30 failed: {e}")
            time.sleep(2)
    else:
        print("Failed to connect to ComfyUI UI after 30 attempts")
    
    # Send a test generation request to the API
    test_prompt = "A surreal landscape painting of mountains under a purple sky, digital art"
    print(f"Requesting image generation for prompt: '{test_prompt}'")
    
    try:
        resp = requests.post(api_url, json={"prompt": test_prompt}, timeout=120)
        if resp.status_code == 200:
            # Save the output image to a file
            output_file = "test_output.png"
            with open(output_file, "wb") as f:
                f.write(resp.content)
            print(f"✅ Image generated successfully and saved to {output_file}")
        else:
            print(f"❌ Generation request failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"❌ Error during generation request: {e}")

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
    print("=" * 60)
    print(f"📱 UI     → {ui_url}")
    print(f"🔗 API    → {api_url}")
    print(f"📊 Status → {status_url}")
    print("=" * 60)
    
    # Test the endpoints
    print("🧪 Running API test...")
    try:
        test_endpoints.remote(ui_url, api_url, status_url)
    except Exception as e:
        print(f"⚠️  Test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Check server status:")
    print(f"   curl {status_url}")
    print("2. Visit the UI URL in your browser")
    print("   - Use ComfyUI credentials: "+CUI_UNAME+" / "+CUI_PASS)
    print("3. Use the API URL directly (no auth required):")
    print(f"   curl -X POST '{api_url}' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"prompt\": \"A beautiful sunset over mountains\"}' \\")
    print("        --output generated_image.png")
    print("4. API is now publicly accessible for development!")
    print("=" * 60)

if __name__ == "__main__":
    main()
