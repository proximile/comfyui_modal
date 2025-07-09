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

# Define a persistent volume for caching Hugging Face downloads (to avoid repeated large downloads)
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Build the container image with required packages and ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # for comfy-cli to clone repos
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.4.1")
    .pip_install("huggingface_hub[hf_transfer]==0.30.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Install ComfyUI using comfy-cli
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia --version 0.3.44")
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

# Define a function to download the SDXL models and set them up in the image
def hf_download():
    from huggingface_hub import hf_hub_download
    
    # Download SDXL base 1.0 model to cache
    base_model_path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        cache_dir="/cache",
    )
    
    # Download SDXL refiner 1.0 model to cache
    refiner_model_path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        filename="sd_xl_refiner_1.0.safetensors",
        cache_dir="/cache",
    )
    
    # Create checkpoints directory and link both models
    subprocess.run(
        "mkdir -p /root/comfy/ComfyUI/models/checkpoints && "
        f"ln -s {base_model_path} /root/comfy/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors && "
        f"ln -s {refiner_model_path} /root/comfy/ComfyUI/models/checkpoints/sd_xl_refiner_1.0.safetensors",
        shell=True, check=True
    )
    
    print("✅ Downloaded and linked SDXL base and refiner models")
    # Prepare the workflow JSON that uses the base model
    # Note: Both base and refiner models are available in checkpoints
    workflow = {
        "0": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        "1": {
            "class_type": "ClipTextEncode",
            "inputs": {"text": "PLACEHOLDER_PROMPT", "clip": ["0", 1]}
        },
        "2": {
            "class_type": "ClipTextEncode",
            "inputs": {"text": "", "clip": ["0", 1]}
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["0", 0],
                "seed": 0,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["1", 0],
                "negative": ["2", 0],
                "latent_image": ["3", 0],
                "denoise": 1.0
            }
        },
        "5": {
            "class_type": "VAEDecode",
            "inputs": {"latent": ["4", 0], "vae": ["0", 2]}
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ComfyUI", "image": ["5", 0]}
        }
    }
    # Write the workflow JSON to the expected path
    Path("/root/workflow_api.json").write_text(json.dumps(workflow))

# Incorporate the model download and workflow setup into the image build
image = image.run_function(
    hf_download,
    volumes={"/cache": vol}
)

# Define the Modal app
app = modal.App(name="comfyui-sdxl-demo", image=image)

# Define the ComfyUI service class with web UI and API endpoints
@app.cls(
    gpu="A100",
    volumes={"/cache": vol},
    scaledown_window=300  # keep container alive for 5 minutes after use
)
@modal.concurrent(max_inputs=5)  # allow a few concurrent API calls in one container
class ComfyService:
    port: int = 8000  # port where ComfyUI will run internally

    @modal.enter()
    def launch_comfy(self):
        """Launch ComfyUI server in background on container start."""
        cmd = f"comfy launch --background -- --listen 0.0.0.0 --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)
        print("ComfyUI server launched in background.")

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
        # Load base workflow and insert the prompt and unique filename prefix
        workflow_path = "/root/workflow_api.json"
        workflow_data = json.loads(Path(workflow_path).read_text())
        # Insert prompt text into the ClipTextEncode node (id "1")
        workflow_data["1"]["inputs"]["text"] = prompt
        # Create a unique prefix for output file
        unique_id = uuid.uuid4().hex
        workflow_data["6"]["inputs"]["filename_prefix"] = unique_id
        # Save this modified workflow to a temp file
        temp_path = f"/root/{unique_id}.json"
        Path(temp_path).write_text(json.dumps(workflow_data))
        # Run the workflow via comfy CLI
        run_cmd = f"comfy run --workflow {temp_path} --wait --timeout 1200"
        subprocess.run(run_cmd, shell=True, check=True)
        # Read the resulting image file
        output_dir = Path("/root/comfy/ComfyUI/output")
        for f in output_dir.iterdir():
            if f.name.startswith(unique_id):
                img_bytes = f.read_bytes()
                return img_bytes
        raise RuntimeError("Output image not found.")

    @modal.fastapi_endpoint(method="POST", label="generate", requires_proxy_auth=True)
    def generate(self, request: Dict):
        """FastAPI endpoint to generate an image from a prompt. Expects JSON: {"prompt": "..."}"""
        prompt_text = request.get("prompt", "")
        image_bytes = self.infer.local(prompt_text)
        from fastapi import Response
        # Return image as JPEG (could also return PNG; here we assume JPEG for size)
        return Response(image_bytes, media_type="image/jpeg")

    @modal.web_server(port, label="comfyui")
    def serve_ui(self):
        """
        Expose the ComfyUI web UI. Navigate to this endpoint in a browser.
        """
        # We could add a simple health check or info here
        return {"status": "ComfyUI UI server is running."}

# Test function that runs within Modal and has access to the proxy-auth secret
@app.function(secrets=[modal.Secret.from_name("proxy-auth")])
def test_endpoints(ui_url: str, api_url: str):
    """Test function that runs within Modal and has access to secrets."""
    import time, requests
    
    print(f"ComfyUI web UI available at: {ui_url}")
    print(f"ComfyUI API available at: {api_url}")
    
    # Access the proxy auth credentials from the secret
    headers = {
        "Modal-Key": os.environ["TOKEN_ID"], 
        "Modal-Secret": os.environ["TOKEN_SECRET"]
    }
    
    # Wait for ComfyUI server to be fully ready
    print("Waiting for ComfyUI server to be ready...")
    for i in range(30):
        try:
            res = requests.get(f"{ui_url}/system_stats", headers=headers, timeout=5)
            if res.status_code == 200:
                print("ComfyUI system_stats:", res.json())
                break
        except Exception as e:
            print(f"Attempt {i+1}/30 failed: {e}")
            time.sleep(2)
    else:
        print("Failed to connect to ComfyUI server after 30 attempts")
        return
    
    # Send a test generation request to the API
    test_prompt = "A surreal landscape painting of mountains under a purple sky."
    print(f"Requesting image generation for prompt: '{test_prompt}'")
    
    try:
        resp = requests.post(api_url, json={"prompt": test_prompt}, headers=headers, timeout=60)
        if resp.status_code == 200:
            # Save the output image to a file
            output_file = "output.jpg"
            with open(output_file, "wb") as f:
                f.write(resp.content)
            print(f"✅ Image generated successfully and saved to {output_file}")
        else:
            print(f"❌ Generation request failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"❌ Error during generation request: {e}")

# Alternative local test function that uses your local Modal authentication
@app.function()
def test_endpoints_without_auth(ui_url: str, api_url: str):
    """Alternative test function that doesn't require proxy auth secrets."""
    import time, requests
    
    print(f"ComfyUI web UI available at: {ui_url}")
    print(f"ComfyUI API available at: {api_url}")
    
    # Note: This won't work with endpoints that require proxy auth, 
    # but can be used to test the UI endpoint
    try:
        res = requests.get(f"{ui_url}/", timeout=5)
        if res.status_code == 200:
            print("✅ UI endpoint is accessible")
        else:
            print(f"UI endpoint returned status {res.status_code}")
    except Exception as e:
        print(f"Failed to connect to UI endpoint: {e}")
    
    print("Note: API endpoint requires proxy auth and cannot be tested without credentials")

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

    print("=" * 60)
    print("🚀 ComfyUI SDXL Service Deployed Successfully!")
    print("📦 Models: SDXL Base 1.0 + Refiner 1.0")
    print("=" * 60)
    print(f"📱 UI  → {ui_url}")
    print(f"🔗 API → {api_url}")
    print("=" * 60)
    
    # Test UI endpoint accessibility
    print("🔍 Testing UI endpoint accessibility...")
    try:
        import requests
        response = requests.get(ui_url, timeout=10)
        if response.status_code == 200:
            print("✅ UI endpoint is accessible without proxy auth")
        else:
            print(f"⚠️  UI endpoint returned status {response.status_code}")
            if response.status_code == 407:
                print("🔒 Proxy auth is required. Checking with Modal auth...")
                # Try to get Modal auth token
                try:
                    from modal.client import Client
                    client = Client()
                    token = client.token
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(ui_url, headers=headers, timeout=10)
                    print(f"With Modal auth: {response.status_code}")
                except Exception as e:
                    print(f"Could not get Modal auth: {e}")
    except Exception as e:
        print(f"Could not test UI endpoint: {e}")
    
    # Run the test function that has access to proxy auth secrets
    try:
        print("\n🧪 Running authenticated API test...")
        test_endpoints.remote(ui_url, api_url)
    except Exception as e:
        print(f"⚠️  Authenticated test failed: {e}")
        print("This might be because the proxy-auth secret is not configured.")
        print("You can still access the endpoints manually with your Modal credentials.")
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Visit the UI URL in your browser")
    print("   - Modal will handle authentication first")
    print("   - Then use ComfyUI credentials: "+CUI_UNAME+" / "+CUI_PASS)
    print("2. Use the API URL with your Modal credentials:")
    print(f"   curl -X POST '{api_url}' \\")
    print("        -H 'Modal-Key: your_token_id' \\")
    print("        -H 'Modal-Secret: your_token_secret' \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"prompt\": \"A beautiful sunset over mountains\"}' \\")
    print("        --output generated_image.jpg")
    print("3. Get your tokens from: https://modal.com/settings/tokens")
    print("=" * 60)

if __name__ == "__main__":
    main()
