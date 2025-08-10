import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
import modal
import base64
import os
import time
import string
import random

VOLUME_NAME = "lora-vol"
VOL = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

USERNAME = "webui"
allchars = string.ascii_letters + string.digits
CUI_PASS = ''.join(random.choice(allchars) for _ in range(8))

# Create Modal app
app = modal.App("i2v-comfyui")

# Define all packages, repos, and files as lists
APT_PACKAGES = [
    "git", "git-lfs", "wget", "curl", "ffmpeg",
    "libsm6", "libxext6", "libxrender-dev",
    "libgomp1", "libglib2.0-0"
]

PIP_PACKAGES = ["hf_transfer", "requests"]

ENVIRONMENT_VARS = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    "CUDA_VISIBLE_DEVICES": "0",
    "TORIO_USE_FFMPEG_VERSION": "5",
    "STARLETTE_KEEPALIVE_TIMEOUT": "300",
    "UVICORN_TIMEOUT_KEEP_ALIVE": "300",
    "HTTP_PROXY_TIMEOUT": "300",
    "COMFYUI_USERNAME": USERNAME,
    "COMFYUI_PASSWORD": CUI_PASS,
}

MODEL_DIRECTORIES = [
    "checkpoints",      # For SDXL models
    "diffusion_models", # For Hunyuan Video models
    "text_encoders", 
    "vae",
    "clip_vision",
    "loras",
    "embeddings",       # For text embeddings
    "clip",            # For CLIP models
    "unet",            # For standalone UNet models
    "flux"             # For Flux models directory
]

# Control video model downloads with environment variable
DOWNLOAD_VIDEO_MODELS = os.getenv("DOWNLOAD_VIDEO_MODELS", "false").lower() == "true"

# Video models (Hunyuan Video and WAN) - large downloads
VIDEO_MODEL_DOWNLOADS = [
    # Hunyuan Video models
    ("Comfy-Org/HunyuanVideo_repackaged", [
        "split_files/diffusion_models/hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors",
        "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors"
    ], "diffusion_models"),
    
    ("Comfy-Org/HunyuanVideo_repackaged", [
        "split_files/vae/hunyuan_video_vae_bf16.safetensors"
    ], "vae"),
    
    ("Comfy-Org/HunyuanVideo_repackaged", [
        "split_files/text_encoders/clip_l.safetensors",
        "split_files/text_encoders/llava_llama3_fp16.safetensors"
    ], "text_encoders"),
    
    ("Comfy-Org/HunyuanVideo_repackaged", [
        "split_files/clip_vision/llava_llama3_vision.safetensors"
    ], "clip_vision"),
    
    ("Kijai/HunyuanVideo_comfy", [
        "hunyuan_video_vae_bf16.safetensors"
    ], "vae"),
    
    # WAN models
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", [
        "split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors"
    ], "diffusion_models"),
    
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", [
        "split_files/text_encoders/umt5_xxl_fp16.safetensors"
    ], "text_encoders"),
    
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", [
        "split_files/vae/wan_2.1_vae.safetensors"
    ], "vae"),
    
    ("Kijai/WanVideo_comfy", [
        "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
    ], "loras"),
]

# Base models (SDXL and essential components) - always downloaded
BASE_MODEL_DOWNLOADS = [
    # SDXL Base Model
    ("stabilityai/stable-diffusion-xl-base-1.0", [
        "sd_xl_base_1.0.safetensors"
    ], "checkpoints"),
    
    # SDXL Refiner Model
    ("stabilityai/stable-diffusion-xl-refiner-1.0", [
        "sd_xl_refiner_1.0.safetensors"
    ], "checkpoints"),
    
    # SDXL Refiner individual components
    ("stabilityai/stable-diffusion-xl-refiner-1.0", [
        "unet/diffusion_pytorch_model.safetensors"
    ], "diffusion_models"),
    
    ("stabilityai/stable-diffusion-xl-refiner-1.0", [
        "text_encoder_2/model.safetensors"
    ], "text_encoders"),
    
    ("stabilityai/stable-diffusion-xl-refiner-1.0", [
        "vae/diffusion_pytorch_model.safetensors"
    ], "vae"),
    
    # SDXL VAE (improved version)
    ("stabilityai/sdxl-vae", [
        "sdxl_vae.safetensors"
    ], "vae"),

    #https://huggingface.co/luisrguerra/realism-refiner-sdxl-real-dream/blob/main/refiner-gold-2.1-rdxl.safetensors
    ("luisrguerra/realism-refiner-sdxl-real-dream", [
        "refiner-gold-2.1-rdxl.safetensors"
    ], "checkpoints"),

    #("John6666/realism-by-stable-yogi-v50fp16-sdxl", [
    #    "diffusion_pytorch_model.safetensors"
    #], "unet"),
]

# Combine models based on settings
HF_DOWNLOADS = BASE_MODEL_DOWNLOADS.copy()
if DOWNLOAD_VIDEO_MODELS:
    HF_DOWNLOADS.extend(VIDEO_MODEL_DOWNLOADS)

# Custom nodes repositories
CUSTOM_NODES = [
    "https://github.com/kijai/ComfyUI-HunyuanVideoWrapper",
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet",
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
    "https://github.com/kijai/ComfyUI-KJNodes",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
    "https://github.com/rgthree/rgthree-comfy",
]

# Build container image with all models included
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*APT_PACKAGES)
    .run_commands("pip install --upgrade pip")
    .run_commands("pip install --upgrade comfy-cli")
    .pip_install(*PIP_PACKAGES)
    .env(ENVIRONMENT_VARS)
    
    # Install ComfyUI
    .run_commands("comfy --skip-prompt install --nvidia --cuda-version 12.6")
    .run_commands("cd /root/comfy/ComfyUI/ && git pull")
    
    # Create model directories
    .run_commands(
        f"mkdir -p /root/comfy/ComfyUI/models/{{{','.join(MODEL_DIRECTORIES)}}}"
    )
)

# Download HuggingFace models (video models only included if DOWNLOAD_VIDEO_MODELS=true)
for repo, files, target_dir in HF_DOWNLOADS:
    for file in files:
        image = image.run_commands(
            f"hf download {repo} "
            f"{file} "
            f"--local-dir /root/comfy/ComfyUI/models/{target_dir}"
        )

# Clone custom nodes
image = image.run_commands("cd /root/comfy/ComfyUI/custom_nodes")
for repo in CUSTOM_NODES:
    image = image.run_commands(
        f"cd /root/comfy/ComfyUI/custom_nodes && git clone {repo}"
    )

# Fix permissions
image = image.run_commands(
    "chmod -R 755 /root/comfy/ComfyUI/custom_nodes",
    "ls -la /root/comfy/ComfyUI/custom_nodes/ComfyUI-HunyuanVideoWrapper/"
)

# Install requirements for each custom node
for repo in CUSTOM_NODES:
    node_name = repo.split('/')[-1]
    image = image.run_commands(
        f"cd /root/comfy/ComfyUI/custom_nodes/{node_name} && "
        f"if [ -f requirements.txt ]; then pip install -r requirements.txt; fi"
    )

# Also install ComfyUI-Manager requirements if it exists
image = image.run_commands(
    "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Manager && "
    "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi"
)

# Update ComfyUI to latest version
image = image.run_commands("cd /root/comfy/ComfyUI/ && git pull && git checkout master")

# Warm up the installation
image = image.run_commands(
    "cd /root/comfy/ComfyUI && python -m compileall . && "
    "python -c 'import torch; torch.cuda.empty_cache()'"
)

# Web server for ComfyUI interface
@app.function(
    image=image,
    gpu="A100-80GB",
    scaledown_window=3000,
    volumes={"/data": VOL},
    memory=32768*4,
    cpu=16.0,
    ephemeral_disk=int(524288*1.5),
    max_containers=1,
)
@modal.web_server(8000, startup_timeout=120*4)
@modal.concurrent(max_inputs=999)
def comfyui_server():
    """Serve ComfyUI web interface with persistent volume for additional models"""
    import os
    import time
    import shutil
    import glob
    
    comfy_root = Path("/root/comfy/ComfyUI")
    output_dir = comfy_root / "output"
    
    # Set up volume structure
    volume_models_dir = Path("/data/models")
    volume_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ComfyUI model subdirectories
    comfy_models = comfy_root / "models"
    comfy_checkpoints = comfy_models / "checkpoints"
    comfy_loras = comfy_models / "loras" 
    comfy_vae = comfy_models / "vae"
    comfy_embeddings = comfy_models / "embeddings"
    comfy_text_encoders = comfy_models / "text_encoders"
    comfy_clip = comfy_models / "clip"
    comfy_clip_vision = comfy_models / "clip_vision"
    comfy_diffusion_models = comfy_models / "diffusion_models"
    comfy_unet = comfy_models / "unet"
    
    # Create specialized lora subdirectories
    loras_trained = comfy_loras / "trained"
    loras_hunyuan = comfy_loras / "hunyuan"
    loras_extra = comfy_loras / "extra"
    loras_trained.mkdir(exist_ok=True)
    loras_hunyuan.mkdir(exist_ok=True)
    loras_extra.mkdir(exist_ok=True)
    
    print("=== Setting up volume symlinks ===")
    
    # 1. Symlink .safetensors files from /data/models/ (root level) - separate Flux models from checkpoints
    print("\n1. Symlinking models from volume root...")
    volume_model_files = list(volume_models_dir.glob("*.safetensors"))
    flux_models = []
    checkpoint_models = []
    
    for model_file in volume_model_files:
        if "flux" in model_file.name.lower():
            flux_models.append(model_file)
        else:
            # All non-Flux models go to checkpoints: SDXL, Pony, Illustrious, CyberRealistic, etc.
            checkpoint_models.append(model_file)
    
    # Symlink ALL checkpoint models to checkpoints/ (SDXL, Pony, Illustrious, etc.)
    print("  1a. Symlinking checkpoint models (SDXL, Pony, Illustrious, CyberRealistic, etc.)...")
    for checkpoint in checkpoint_models:
        link_path = comfy_checkpoints / checkpoint.name
        if not link_path.exists():
            try:
                link_path.symlink_to(checkpoint)
                print(f"    ✓ {checkpoint.name}")
            except Exception as e:
                print(f"    ✗ Failed to link {checkpoint.name}: {e}")
    print(f"    Total: {len(checkpoint_models)} checkpoint files linked")
    
    # Symlink Flux models to diffusion_models/
    print("  1b. Symlinking Flux models...")
    comfy_diffusion_models.mkdir(exist_ok=True)
    for flux_model in flux_models:
        link_path = comfy_diffusion_models / flux_model.name
        if not link_path.exists():
            try:
                link_path.symlink_to(flux_model)
                print(f"    ✓ {flux_model.name}")
            except Exception as e:
                print(f"    ✗ Failed to link {flux_model.name}: {e}")
    print(f"    Total: {len(flux_models)} Flux model files linked")
    
    # 2. Symlink trained LoRAs from /data/runs/*/checkpoints/*.safetensors to loras/trained/
    print("\n2. Symlinking trained LoRAs...")
    runs_pattern = "/data/runs/*/checkpoints/*.safetensors"
    trained_loras = glob.glob(runs_pattern)
    for lora_path in trained_loras:
        lora_file = Path(lora_path)
        # Create unique name: run_name + original filename
        run_name = lora_file.parent.parent.name
        unique_name = f"{run_name}_{lora_file.name}"
        link_path = loras_trained / unique_name
        
        if not link_path.exists():
            try:
                link_path.symlink_to(lora_file)
                print(f"  ✓ {unique_name}")
            except Exception as e:
                print(f"  ✗ Failed to link {unique_name}: {e}")
    print(f"  Total: {len(trained_loras)} trained LoRA files linked")
    
    # 3. Symlink VAE files from /data/models/vae/ to vae/
    print("\n3. Symlinking VAE files...")
    volume_vae_dir = volume_models_dir / "vae"
    if volume_vae_dir.exists():
        vae_files = list(volume_vae_dir.glob("*.safetensors"))
        for vae_file in vae_files:
            link_path = comfy_vae / vae_file.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(vae_file)
                    print(f"  ✓ {vae_file.name}")
                except Exception as e:
                    print(f"  ✗ Failed to link {vae_file.name}: {e}")
        print(f"  Total: {len(vae_files)} VAE files linked")
    else:
        print("  No VAE directory found in volume")
    
    # 4. Symlink Hunyuan LoRAs from /data/models/hunyuan_loras/ to loras/hunyuan/
    print("\n4. Symlinking Hunyuan LoRAs...")
    volume_hunyuan_loras = volume_models_dir / "hunyuan_loras"
    if volume_hunyuan_loras.exists():
        hunyuan_files = list(volume_hunyuan_loras.glob("*.safetensors"))
        for hunyuan_file in hunyuan_files:
            link_path = loras_hunyuan / hunyuan_file.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(hunyuan_file)
                    print(f"  ✓ {hunyuan_file.name}")
                except Exception as e:
                    print(f"  ✗ Failed to link {hunyuan_file.name}: {e}")
        print(f"  Total: {len(hunyuan_files)} Hunyuan LoRA files linked")
    else:
        print("  No Hunyuan LoRAs directory found in volume")
    
    # 5. Symlink extra LoRAs from /data/models/loras/ to loras/extra/
    print("\n5. Symlinking extra LoRAs from volume...")
    volume_loras_dir = volume_models_dir / "loras"
    if volume_loras_dir.exists():
        lora_files = list(volume_loras_dir.glob("*.safetensors"))
        for lora_file in lora_files:
            link_path = loras_extra / lora_file.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(lora_file)
                    print(f"  ✓ {lora_file.name}")
                except Exception as e:
                    print(f"  ✗ Failed to link {lora_file.name}: {e}")
        print(f"  Total: {len(lora_files)} extra LoRA files linked")
    else:
        print("  No loras directory found in volume")
    
    # 6. Symlink embeddings from /data/models/embeddings/ to embeddings/
    print("\n6. Symlinking embeddings...")
    volume_embeddings_dir = volume_models_dir / "embeddings"
    if volume_embeddings_dir.exists():
        # Create embeddings directory if it doesn't exist
        comfy_embeddings.mkdir(exist_ok=True)
        
        # Symlink all files in embeddings directory (safetensors, pt, bin files)
        embedding_files = list(volume_embeddings_dir.glob("*"))
        linked_count = 0
        for embedding_item in embedding_files:
            if embedding_item.is_file():
                link_path = comfy_embeddings / embedding_item.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(embedding_item)
                        print(f"  ✓ {embedding_item.name}")
                        linked_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to link {embedding_item.name}: {e}")
            elif embedding_item.is_dir():
                # Handle subdirectories by creating them and symlinking their contents
                subdir_path = comfy_embeddings / embedding_item.name
                subdir_path.mkdir(exist_ok=True)
                for subfile in embedding_item.glob("*"):
                    if subfile.is_file():
                        link_path = subdir_path / subfile.name
                        if not link_path.exists():
                            try:
                                link_path.symlink_to(subfile)
                                print(f"  ✓ {embedding_item.name}/{subfile.name}")
                                linked_count += 1
                            except Exception as e:
                                print(f"  ✗ Failed to link {embedding_item.name}/{subfile.name}: {e}")
        print(f"  Total: {linked_count} embedding files linked")
    else:
        print("  No embeddings directory found in volume")
    
    # 7. Symlink Flux text encoders from /data/models/flux_text_encoders/ to text_encoders/
    print("\n7. Symlinking Flux text encoders...")
    volume_flux_text_encoders = volume_models_dir / "flux_text_encoders"
    if volume_flux_text_encoders.exists():
        # Create text_encoders directory if it doesn't exist
        comfy_text_encoders.mkdir(exist_ok=True)
        
        flux_encoder_files = list(volume_flux_text_encoders.glob("*"))
        linked_count = 0
        for encoder_file in flux_encoder_files:
            if encoder_file.is_file():
                link_path = comfy_text_encoders / encoder_file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(encoder_file)
                        print(f"  ✓ {encoder_file.name}")
                        linked_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to link {encoder_file.name}: {e}")
        print(f"  Total: {linked_count} Flux text encoder files linked")
    else:
        print("  No flux_text_encoders directory found in volume")
    
    # 8. Symlink Flux VAE from /data/models/flux_vae/ to vae/
    print("\n8. Symlinking Flux VAE files...")
    volume_flux_vae = volume_models_dir / "flux_vae"
    if volume_flux_vae.exists():
        flux_vae_files = list(volume_flux_vae.glob("*"))
        linked_count = 0
        for vae_file in flux_vae_files:
            if vae_file.is_file():
                link_path = comfy_vae / vae_file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(vae_file)
                        print(f"  ✓ {vae_file.name}")
                        linked_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to link {vae_file.name}: {e}")
        print(f"  Total: {linked_count} Flux VAE files linked")
    else:
        print("  No flux_vae directory found in volume")
    
    # 9. Symlink CLIP models from /data/models/clip/ to clip/
    print("\n9. Symlinking CLIP models...")
    volume_clip_dir = volume_models_dir / "clip"
    if volume_clip_dir.exists():
        # Create clip directory if it doesn't exist
        comfy_clip.mkdir(exist_ok=True)
        
        clip_files = list(volume_clip_dir.glob("*"))
        linked_count = 0
        for clip_file in clip_files:
            if clip_file.is_file():
                link_path = comfy_clip / clip_file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(clip_file)
                        print(f"  ✓ {clip_file.name}")
                        linked_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to link {clip_file.name}: {e}")
        print(f"  Total: {linked_count} CLIP model files linked")
    else:
        print("  No clip directory found in volume")
    
    # 10. Symlink CLIP vision models from /data/models/clip_vision/ to clip_vision/
    print("\n10. Symlinking CLIP vision models...")
    volume_clip_vision_dir = volume_models_dir / "clip_vision"
    if volume_clip_vision_dir.exists():
        # Create clip_vision directory if it doesn't exist
        comfy_clip_vision.mkdir(exist_ok=True)
        
        clip_vision_files = list(volume_clip_vision_dir.glob("*"))
        linked_count = 0
        for clip_vision_file in clip_vision_files:
            if clip_vision_file.is_file():
                link_path = comfy_clip_vision / clip_vision_file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(clip_vision_file)
                        print(f"  ✓ {clip_vision_file.name}")
                        linked_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to link {clip_vision_file.name}: {e}")
        print(f"  Total: {linked_count} CLIP vision model files linked")
    else:
        print("  No clip_vision directory found in volume")
    
    # 11. Symlink diffusion models from /data/models/diffusion_models/ to diffusion_models/
    print("\n11. Symlinking diffusion models...")
    volume_diffusion_models_dir = volume_models_dir / "diffusion_models"
    if volume_diffusion_models_dir.exists():
        # Create diffusion_models directory if it doesn't exist
        comfy_diffusion_models.mkdir(exist_ok=True)
        
        diffusion_files = list(volume_diffusion_models_dir.glob("*"))
        linked_count = 0
        for diffusion_file in diffusion_files:
            if diffusion_file.is_file():
                link_path = comfy_diffusion_models / diffusion_file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(diffusion_file)
                        print(f"  ✓ {diffusion_file.name}")
                        linked_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to link {diffusion_file.name}: {e}")
        print(f"  Total: {linked_count} diffusion model files linked")
    else:
        print("  No diffusion_models directory found in volume")
    
    os.chdir("/root/comfy/ComfyUI")
    
    
    # Set up output directory with volume
    print("\n=== Setting up output directory ===")
    if output_dir.exists() or output_dir.is_symlink():
        print(f"Removing existing output directory or symlink: {output_dir}")
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    sample_outputs = Path("/data/sample_outputs")
    sample_outputs.mkdir(parents=True, exist_ok=True)
    print(f"Creating output symlink: {output_dir} → {sample_outputs}")
    output_dir.symlink_to(sample_outputs, target_is_directory=True)

    # List available models for debugging
    print("\n=== Model Summary ===")
    
    # Built-in models
    print("\nBuilt-in models:")
    for model_dir in ["checkpoints", "diffusion_models", "vae", "text_encoders", "clip", "clip_vision"]:
        model_path = comfy_root / "models" / model_dir
        if model_path.exists():
            models = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.ckpt"))
            builtin_models = [m for m in models if not m.is_symlink()]
            total_models = len(models)
            volume_models = total_models - len(builtin_models)
            print(f"  {model_dir}: {len(builtin_models)} built-in + {volume_models} from volume = {total_models} total")
    
    # LoRA structure
    print("\nLoRA structure:")
    loras_base = comfy_root / "models" / "loras"
    if loras_base.exists():
        builtin_loras = list(loras_base.glob("*.safetensors"))
        print(f"  Built-in LoRAs: {len(builtin_loras)}")
        
        if loras_trained.exists():
            trained_count = len(list(loras_trained.glob("*.safetensors")))
            print(f"  Trained LoRAs: {trained_count}")
        
        if loras_hunyuan.exists():
            hunyuan_count = len(list(loras_hunyuan.glob("*.safetensors")))
            print(f"  Hunyuan LoRAs: {hunyuan_count}")
        
        if loras_extra.exists():
            extra_count = len(list(loras_extra.glob("*.safetensors")))
            print(f"  Extra LoRAs: {extra_count}")
    
    # Volume checkpoints
    volume_checkpoint_count = len(list(comfy_checkpoints.glob("*.safetensors")))
    builtin_checkpoint_count = len([f for f in comfy_checkpoints.glob("*.safetensors") if not f.is_symlink()])
    volume_linked_checkpoints = volume_checkpoint_count - builtin_checkpoint_count
    print(f"\nCheckpoints: {builtin_checkpoint_count} built-in + {volume_linked_checkpoints} from volume = {volume_checkpoint_count} total")
    
    # VAE files
    vae_total = len(list(comfy_vae.glob("*.safetensors")))
    vae_builtin = len([f for f in comfy_vae.glob("*.safetensors") if not f.is_symlink()])
    vae_volume = vae_total - vae_builtin
    print(f"VAE files: {vae_builtin} built-in + {vae_volume} from volume = {vae_total} total")

    # Start ComfyUI server
    subprocess.Popen([
        "python", "main.py",
        "--listen", "0.0.0.0",
        "--port", "8000",
        "--highvram",
        "--use-pytorch-cross-attention",
        "--enable-cors-header",
        "--verbose",
        "--preview-method", "auto"
    ])

# Function to upload models to volume
@app.function(
    image=image,
    volumes={"/data": VOL},
    timeout=3600,
)
def upload_model_to_volume(model_data: bytes, filename: str, destination: str = "models"):
    """
    Upload a model file to the persistent volume
    
    Args:
        model_data: The model file as bytes
        filename: Name of the file (e.g., "my_model.safetensors")
        destination: Destination path relative to /data/ 
                    (e.g., "models", "models/vae", "models/hunyuan_loras")
    """
    volume_path = Path(f"/data/{destination}")
    volume_path.mkdir(parents=True, exist_ok=True)
    
    file_path = volume_path / filename
    with open(file_path, "wb") as f:
        f.write(model_data)
    
    # Commit changes to volume
    VOL.commit()
    
    print(f"Uploaded {filename} to volume at {file_path}")
    return str(file_path)

# Function to list models in volume
@app.function(
    image=image,
    volumes={"/data": VOL},
)
def list_volume_models(path: str = "models"):
    """
    List all models in the volume for a given path
    
    Args:
        path: Path relative to /data/ (e.g., "models", "models/vae", "runs")
    """
    volume_path = Path(f"/data/{path}")
    
    if not volume_path.exists():
        return {}
    
    result = {
        "path": str(volume_path),
        "files": [],
        "subdirs": []
    }
    
    # List files
    for ext in ["*.safetensors", "*.ckpt", "*.pt", "*.pth"]:
        for model_file in volume_path.glob(ext):
            if model_file.is_file():
                result["files"].append({
                    "name": model_file.name,
                    "size": model_file.stat().st_size,
                    "path": str(model_file)
                })
    
    # List subdirectories
    for item in volume_path.iterdir():
        if item.is_dir():
            result["subdirs"].append(item.name)
    
    return result

# Function to create training run directory structure
@app.function(
    image=image,
    volumes={"/data": VOL},
)
def create_training_run(run_name: str):
    """
    Create a new training run directory structure
    
    Args:
        run_name: Name of the training run
    """
    run_path = Path(f"/data/runs/{run_name}")
    checkpoints_path = run_path / "checkpoints"
    logs_path = run_path / "logs"
    
    # Create directory structure
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    VOL.commit()
    
    return {
        "run_path": str(run_path),
        "checkpoints_path": str(checkpoints_path),
        "logs_path": str(logs_path)
    }

# Function to list all training runs
@app.function(
    image=image,
    volumes={"/data": VOL},
)
def list_training_runs():
    """List all training runs and their contents"""
    runs_path = Path("/data/runs")
    
    if not runs_path.exists():
        return {}
    
    runs = {}
    for run_dir in runs_path.iterdir():
        if run_dir.is_dir():
            checkpoints_dir = run_dir / "checkpoints"
            runs[run_dir.name] = {
                "path": str(run_dir),
                "checkpoints": []
            }
            
            if checkpoints_dir.exists():
                for checkpoint in checkpoints_dir.glob("*.safetensors"):
                    runs[run_dir.name]["checkpoints"].append({
                        "name": checkpoint.name,
                        "path": str(checkpoint),
                        "size": checkpoint.stat().st_size
                    })
    return runs


@app.local_entrypoint()
def main():
    print(f"👤 Username: {USERNAME}")
    print(f"🔑 Password: {CUI_PASS}")
    print("="*70)
    print("\n📝 Notes:")
    print("- Basic auth is handled by the ComfyUI custom node")
    print("- Models are symlinked from /data/models/")
    print("- LoRAs are symlinked from /data/runs/*/checkpoints/")
    print("- Outputs are saved to /data/sample_outputs/")
    print("="*70)

if __name__ == "__main__":
    main()
