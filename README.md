# ComfyUI-MacOffload-TE ⚡

> **High-Performance Cross-Platform Text Encoder Offloading for ComfyUI**  
> Offload heavy Text Encoders (Flux.2 Klein, Qwen-Image-2.1) from your VRAM-constrained GPU (e.g. RTX 4060 8GB) to a macOS machine with Apple Silicon Unified Memory (MPS) over your local network — eliminating model swapping, boosting generation speed, and unlocking higher native image resolutions.

---

## 1. Overview & Motivation

When running state-of-the-art diffusion pipelines (such as Flux Klein or Qwen Image) on consumer GPUs with 8GB VRAM (such as RTX 4060 8GB), the text encoder often occupies 8GB to 18GB of memory. On single-GPU setups, this forces ComfyUI into continuous, aggressive model swapping (VRAM thrashing between GPU and system RAM over PCIe), causing steep performance degradation, sluggish iteration cycles, and frequent out-of-memory (OOM) crashes.

**ComfyUI-MacOffload-TE** eliminates this bottleneck by offloading heavy text encoders to a remote macOS device (such as a MacBook, Mac mini, or Mac Studio) equipped with Apple Silicon Unified Memory:

- **0 MB Local VRAM Footprint**: The entire text encoder runs remotely on Apple Silicon MPS, completely freeing your local GPU from text encoder memory overhead.
- **Faster Generation & Higher Efficiency**: Completely bypasses the sluggish PCIe model swapping phase between text encoding and diffusion sampling. The local GPU stays primed with diffusion weights, launching sampling immediately and dramatically accelerating generation throughput.
- **Higher Direct Output Resolutions**: Reclaiming precious VRAM headroom allows the local GPU to allocate maximum memory to the diffusion transformer (DiT) and VAE, enabling direct generation at substantially higher native image resolutions and larger latent dimensions without OOM errors.
- **Dual-Device Synergy**:
  - **Client (Windows)**: Dedicates 100% of GPU compute and VRAM to diffusion backbone sampling and VAE decoding.
  - **Server (macOS / Apple Silicon)**: Keeps large text encoders resident in Unified Memory (MPS), serving tokenization and embedding forward passes over low-latency binary tensor streaming.

---

## 2. Architecture & Communication Flow

### Text Architecture Overview
```text
+-------------------------------------------------------------+
|                      Windows Client                         |
|  [ComfyUI]                                                  |
|   └── Custom Node (Facade + Helpers)                        |
|        ├── Pack prompt text & optional reference images     |
|        ├── Serialize to PyTorch binary stream (in-memory)   |
|        └── HTTP POST (application/octet-stream)             |
+------------------------------+------------------------------+
                               |
                   Local Area Network (LAN)
                   Low-latency Binary Stream
                               |
+------------------------------v------------------------------+
|                     macOS Server (MPS)                      |
|  [FastAPI Bridge]                                           |
|   ├── Route: /v1/encode/{model_name}                        |
|   ├── Dynamic Model Pool (Auto-discovery + LRU Eviction)    |
|   ├── Direct Accelerated MPS Loading (safetensors mmap)     |
|   └── Return serialized Conditioning & Latent tensors       |
+-------------------------------------------------------------+
```

### Flow Sequence
```mermaid
sequenceDiagram
    autonumber
    participant Win as Windows ComfyUI Client
    participant Srv as macOS FastAPI Server
    participant MPS as Apple Silicon MPS Pool

    Win->>Srv: POST /v1/encode/{model_name} (Binary Payload)
    Note over Srv: Dynamic Model Manager checks resident pool
    alt Model Not In Unified Memory
        Srv->>MPS: Direct accelerated load into Unified Memory (bfloat16)
    end
    Srv->>MPS: Execute Forward Pass (Tokenization & Embeddings)
    MPS-->>Srv: Return Conditioning / Latent Tensors
    Srv-->>Win: HTTP 200 (Serialized PyTorch Binary Stream)
    Note over Win: Inject conditioning into diffusion sampling
```

---

## 3. Supported Models

| Model Folder Name | Base Architecture | Hugging Face Repository | Summary |
| :--- | :--- | :--- | :--- |
| `flux2-klein-4b-uncensored-text-encoder` | Qwen3-4B | [`ponpoke/flux2-klein-4b-uncensored-text-encoder`](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder) | Uncensored text encoder variant for FLUX.2-klein 4B DiT. Processed to remove refusal vectors to prevent safety filter refusals. |
| `flux2-klein-9b-uncensored-text-encoder` | Qwen3-8B | [`ponpoke/flux2-klein-9b-uncensored-text-encoder`](https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder) | Uncensored text encoder variant for FLUX.2-klein 9B DiT with higher parameter capacity for nuanced prompt comprehension without safety filter refusals. |
| `Qwen-Image-2.1-Text-Encoder-Heretic` | Qwen3-VL-8B | [`pottokao/Qwen-Image-2.1-Text-Encoder-Heretic`](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic) | Abliterated "Heretic" version of the Qwen-Image-2.1 text encoder based on the Qwen3-VL-8B architecture. Removes safety alignment refusals using directional ablation on `o_proj` and `down_proj` layers while preserving functional vision-language performance. |

---

## 4. Installation & Setup

### 4.1 Client Setup (Windows ComfyUI)

1. Clone or copy this repository into your ComfyUI `custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repo-url> ComfyUI-MacOffload-TE
   ```

2. Restart ComfyUI. The nodes will be available under category:
   `⚡ MacOffload/Conditioning`

3. **Node Controls & UI Features**:
   - **🔄 Fetch Models from Server**: Click this button on any MacOffload node to query the Mac server's `./models` directory and populate the `model_name` dropdown with available models.
   - **Saved Workflow Memory**: The node remembers the previously selected model. When loading an existing workflow, the saved model name is preserved automatically without needing to click fetch again.
   - **❓ Help Badge**: Click the `?` icon in the node title bar (or right-click the node and select **"About this node ❓"**) to view full parameter descriptions, tooltips, and server verification hints.

---

### 4.2 Server Setup (macOS / Apple Silicon)

1. Navigate to the `server/` directory on your Mac:
   ```bash
   cd ComfyUI-MacOffload-TE/server
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install server dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure model paths:
   Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to point `MODELS_DIR` to your model directory (defaults to `~/models`):
   ```ini
   MODELS_DIR=/Users/<your-username>/models
   BRIDGE_PORT=8288
   MAX_MODELS=1
   ```

   Place your model folders inside `MODELS_DIR`:
   ```text
   ~/models/
   ├── flux2-klein-4b-uncensored-text-encoder/
   ├── flux2-klein-9b-uncensored-text-encoder/
   └── Qwen-Image-2.1-Text-Encoder-Heretic/
   ```

5. Launch the server:
   ```bash
   python server.py
   ```

---

## 5. Server Startup Arguments & Options

The server supports command-line arguments, `.env` file, and environment variables:

| CLI Option | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `BRIDGE_HOST` | `0.0.0.0` | Server host IP address to bind to |
| `--port` | `BRIDGE_PORT` | `8288` | Port number to listen on |
| `--max-models` | `MAX_MODELS` | `1` | Maximum concurrent models kept in Apple Silicon Unified Memory (LRU eviction) |
| `--idle-timeout` | `IDLE_TIMEOUT` | `0` | Seconds of inactivity before auto-unloading a model (`0` = keep resident indefinitely) |
| `--models-dir` | `MODELS_DIR` | `~/models` | Root directory to scan for model folders |
| `--device` | `BRIDGE_DEVICE` | `mps` | Target inference device (`mps` on Apple Silicon, `cuda`, or `cpu`) |
| `--settling-delay` | `SETTLING_DELAY` | `1.0` | Memory settling window in seconds after model eviction before loading next model |

### Example CLI Commands:
```bash
# Run with default configuration (.env or ~/models)
python server.py

# Custom port and custom models directory
python server.py --port 8288 --models-dir /Volumes/ExternalSSD/models

# Keep at most 1 resident model with 300s idle auto-unload
python server.py --max-models 1 --idle-timeout 300
```

---

## 6. Node Reference

### 1. `Remote CLIP Text Encode (Flux Klein) ⚡`
- **Class Identifier**: `MACOFFLOAD_RemoteCLIPTextEncode`
- **Category**: `⚡ MacOffload/Conditioning`
- **Description**: Drop-in replacement for native CLIPTextEncode in Flux.2 Klein workflows.
- **Inputs**:
  - `server_url` *(STRING)*: Base URL of the macOS server (e.g. `http://192.168.1.100:8288`).
  - `model_name` *(COMBO)*: Target model folder name (e.g. `flux2-klein-4b-uncensored-text-encoder`).
  - `text` *(STRING)*: Positive or negative text prompt.
- **Outputs**:
  - `conditioning` *(CONDITIONING)*: Formatted conditioning tuple containing stacked hidden states [batch, seq_len, 12288] and pooled representation.

---

### 2. `Remote Text Encode Qwen Image 2.1 ⚡`
- **Class Identifier**: `MACOFFLOAD_RemoteTextEncodeQwenImage21`
- **Category**: `⚡ MacOffload/Conditioning`
- **Description**: Multimodal text and multi-image reference encoder for Qwen-Image-2.1.
- **Inputs**:
  - `server_url` *(STRING)*: Base URL of the macOS server.
  - `model_name` *(COMBO)*: Target model folder name (e.g. `Qwen-Image-2.1-Text-Encoder-Heretic`).
  - `prompt` *(STRING)*: Positive text prompt.
  - `negative_prompt` *(STRING)*: Negative text prompt.
  - `resolution` *(INT, default: 1024)*: Target dimension for reference image scaling (multiples of 32).
  - `vae` *(VAE, optional)*: WanVAE model used to encode reference images into reference latents.
  - `image_1` ~ `image_16` *(IMAGE, optional)*: Up to 16 input reference images.
- **Outputs**:
  - `positive` *(CONDITIONING)*: Positive conditioning with sliced hidden states and vision slots.
  - `negative` *(CONDITIONING)*: Negative conditioning.
  - `latent` *(LATENT)*: WanVAE-compatible 64-channel latent tensor `[1, 64, H//16, W//16]`.

---

## 7. Server REST API Reference

### List Available Models
- **Endpoint**: `GET /v1/models`
- **Description**: Scans `MODELS_DIR` and returns discovered model folder names, their encoder categories, and resident status.
- **Response**:
  ```json
  {
    "models": [
      "flux2-klein-4b-uncensored-text-encoder",
      "flux2-klein-9b-uncensored-text-encoder",
      "Qwen-Image-2.1-Text-Encoder-Heretic"
    ],
    "model_types": {
      "flux2-klein-4b-uncensored-text-encoder": "flux_klein",
      "flux2-klein-9b-uncensored-text-encoder": "flux_klein",
      "Qwen-Image-2.1-Text-Encoder-Heretic": "qwen_image"
    },
    "resident_models": ["flux2-klein-4b-uncensored-text-encoder"],
    "models_dir": "/Users/user/models"
  }
  ```

### Health Probe
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "online",
    "device": "mps",
    "dtype": "torch.bfloat16",
    "max_models": 1,
    "idle_timeout": 0,
    "models_dir": "/Users/user/models",
    "active_models": ["flux2-klein-4b-uncensored-text-encoder"],
    "discovered_models": [
      "flux2-klein-4b-uncensored-text-encoder",
      "flux2-klein-9b-uncensored-text-encoder",
      "Qwen-Image-2.1-Text-Encoder-Heretic"
    ]
  }
  ```

### Dynamic Encoding
- **Endpoint**: `POST /v1/encode/{model_name}`
- **Headers**: `Content-Type: application/octet-stream`
- **Body**: In-memory PyTorch serialized binary dictionary containing prompt strings and tensor lists.
- **Response**: Serialized PyTorch binary dictionary matching ComfyUI output contracts.

### Manual Memory Eviction
- **Endpoint**: `POST /v1/unload`
- **Body (optional JSON)**:
  ```json
  {"model_name": "flux2-klein-4b-uncensored-text-encoder"}
  ```
  *(Omit `model_name` or send `{}` to evict all loaded models and flush Unified Memory).*

---

## 8. Troubleshooting & FAQ

### Q: Node fails with `Connection failed to 'http://...:8288'`
1. Check if the Mac server is running (`python server.py`).
2. Verify both machines are on the same local network subnet.
3. Verify the IP address by running `ipconfig` (Windows) and `ifconfig` or `ipconfig getifaddr en0` (macOS).
4. Check macOS Firewall (*System Settings -> Network -> Firewall*): ensure incoming connections to Python are allowed.
5. Test connectivity from Windows PowerShell:
   ```powershell
   curl http://<mac-ip>:8288/health
   ```

### Q: How does model selection work?
- The node remembers the model name from your last run or saved workflow.
- To discover new models added to your Mac server's models directory, click the **"🔄 Fetch Models from Server"** button on the node. The dropdown list will update immediately while retaining your current selection if still valid.

---

## 9. License

MIT License. Free for personal and commercial workflows.
