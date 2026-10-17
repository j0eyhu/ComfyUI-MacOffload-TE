# server.py
import argparse
import asyncio
import json
import os
import struct
import sys
import traceback
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Prevent PyTorch MPS allocator from hanging/throttling on unified memory during model swaps
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import safetensors.torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse


def deserialize_safetensors_request(buf: bytes):
    if len(buf) < 8:
        raise ValueError("Payload buffer too short for safetensors.")
    header_size = struct.unpack("<Q", buf[:8])[0]
    if len(buf) < 8 + header_size:
        raise ValueError("Malformed safetensors buffer: truncated header.")
    header_raw = json.loads(buf[8 : 8 + header_size].decode("utf-8"))
    raw_meta = header_raw.get("__metadata__", {})
    meta = json.loads(raw_meta.get("meta", "{}"))
    tensors = safetensors.torch.load(buf)
    return tensors, meta


def serialize_safetensors_response(tensors: dict, meta: dict) -> bytes:
    clean_tensors = {
        k: v.detach().cpu().contiguous()
        for k, v in tensors.items()
        if isinstance(v, torch.Tensor)
    }
    wrapped_meta = {"meta": json.dumps(meta)}
    return safetensors.torch.save(clean_tensors, metadata=wrapped_meta)

# Ensure current server directory is on sys.path regardless of execution working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from model_manager import DynamicModelManager

# Parse CLI arguments with fallback to config defaults
parser = argparse.ArgumentParser(description="ComfyUI Condition Bridge Server")
parser.add_argument(
    "--max-models",
    type=int,
    default=None,
    help=f"Maximum number of models resident in unified memory (default: {config.MAX_MODELS})"
)
parser.add_argument(
    "--idle-timeout",
    type=int,
    default=None,
    help=f"Idle timeout in seconds before auto-unloading (default: {config.IDLE_TIMEOUT}, 0=disabled)"
)
parser.add_argument(
    "--port",
    type=int,
    default=None,
    help=f"Server listen port (default: {config.PORT})"
)
parser.add_argument(
    "--host",
    type=str,
    default=None,
    help=f"Server listen host (default: {config.HOST})"
)
parser.add_argument(
    "--models-dir",
    type=str,
    default=None,
    help=f"Root directory containing model folders (default: {config.MODELS_DIR})"
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help=f"Inference device (default: {config.DEVICE})"
)
parser.add_argument(
    "--settling-delay",
    type=float,
    default=None,
    help=f"Memory settling window in seconds after model eviction (default: {config.SETTLING_DELAY})"
)
args, _ = parser.parse_known_args()

# Apply runtime CLI overrides
config.update_runtime_config(
    host=args.host,
    port=args.port,
    max_models=args.max_models,
    idle_timeout=args.idle_timeout,
    device=args.device,
    models_dir=args.models_dir,
    settling_delay=args.settling_delay,
)

# Initialize pool manager
model_manager = DynamicModelManager(
    max_models=config.MAX_MODELS,
    idle_timeout=config.IDLE_TIMEOUT,
    settling_delay=config.SETTLING_DELAY
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages server lifecycle, starting watchdog and ensuring clean shutdown.
    """
    print(
        f"[Server Startup] DynamicModelManager initialized "
        f"(max_resident={config.MAX_MODELS}, idle_timeout={config.IDLE_TIMEOUT}s, "
        f"settling_delay={config.SETTLING_DELAY}s, device={config.DEVICE}, "
        f"models_dir='{config.MODELS_DIR}')."
    )
    model_manager.start_watchdog()
    yield
    print("[Server Shutdown] Cleaning up all models and background tasks...")
    model_manager.stop_watchdog()
    await model_manager.unload_all()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ComfyUI Condition Bridge",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin browser requests (e.g. from ComfyUI Web UI)
# Using allow_credentials=False complies with W3C wildcard origin specs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def add_private_network_headers(request: Request, call_next):
    """
    Support Chromium Private Network Access (PNA) specifications for LAN queries.
    """
    response = await call_next(request)
    if "access-control-request-private-network" in request.headers:
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.get("/health")
async def health_check():
    """
    Service health probe and resident pool inspection.
    """
    model_manager.scan_models()
    return JSONResponse(
        content={
            "status": "online",
            "device": config.DEVICE,
            "dtype": str(config.DTYPE),
            "max_models": model_manager.max_models,
            "idle_timeout": model_manager.idle_timeout,
            "active_models": list(model_manager.pool.keys()),
            "discovered_models": list(model_manager.discovered_models.keys()),
        }
    )


@app.get("/v1/models")
async def list_models():
    """
    Returns list of discovered model folder names and their encoder categories.
    Used by ComfyUI client node to populate the dynamic model selection dropdown.
    """
    discovered = model_manager.scan_models()
    models = list(discovered.keys())
    model_types = {name: info["type"] for name, info in discovered.items()}
    return JSONResponse(
        content={
            "models": models,
            "model_types": model_types,
            "resident_models": list(model_manager.pool.keys()),
        }
    )


@app.post("/v1/encode/{model_name}")
async def encode_dynamic(model_name: str, request: Request):
    """
    Generic endpoint routing requests to the specified registered encoder using
    safe, non-executable safetensors binary streams.
    """
    model_name = os.path.basename(model_name.strip())
    try:
        body = await request.body()
        tensors, meta = deserialize_safetensors_request(body)

        # Acquire loaded encoder from pool
        encoder = await model_manager.get_encoder(model_name)

        model_type = meta.get("model_type")
        encoder_name = type(encoder).__name__.lower()

        if model_type in ("flux_klein", "flux3_klein") or "flux" in encoder_name:
            prompt = meta.get("prompt", "")
            result = await asyncio.to_thread(encoder.encode, prompt=prompt)

            cond = result["conditioning"][0][0].contiguous()
            pooled = result["conditioning"][0][1]["pooled_output"].contiguous()
            resp_tensors = {
                "conditioning": cond,
                "pooled_output": pooled,
            }
            resp_meta = {"type": model_type or ("flux3_klein" if "flux3" in encoder_name else "flux_klein")}

        elif model_type == "qwen_image" or "qwen" in encoder_name:
            prompt = meta.get("prompt", "")
            negative_prompt = meta.get("negative_prompt", "")
            keep_vision = meta.get("keep_vision", True)
            image_count = meta.get("image_count", 0)
            images_vl = [
                tensors[f"image_{i}"]
                for i in range(image_count)
                if f"image_{i}" in tensors
            ]

            result = await asyncio.to_thread(
                encoder.encode,
                prompt=prompt,
                negative_prompt=negative_prompt,
                images_vl=images_vl,
                keep_vision=keep_vision,
            )

            pos_cond = result["positive"]
            neg_cond = result["negative"]

            resp_tensors = {
                "positive_embeds": pos_cond[0][0].contiguous(),
                "negative_embeds": neg_cond[0][0].contiguous(),
            }

            pos_extra = pos_cond[0][1] if len(pos_cond[0]) > 1 else {}
            neg_extra = neg_cond[0][1] if len(neg_cond[0]) > 1 else {}

            if "attention_mask" in pos_extra:
                resp_tensors["positive_mask"] = pos_extra["attention_mask"].contiguous()
            if "attention_mask" in neg_extra:
                resp_tensors["negative_mask"] = neg_extra["attention_mask"].contiguous()

            resp_meta = {
                "type": "qwen_image",
                "pos_slots": pos_extra.get("image_slots", []),
                "neg_slots": neg_extra.get("image_slots", []),
            }
        else:
            raise ValueError(f"Unknown or unsupported encoder type: {type(encoder)}")

        response_bytes = serialize_safetensors_response(resp_tensors, resp_meta)
        return Response(
            content=response_bytes,
            media_type="application/octet-stream"
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print("\n" + "!" * 60)
        print(f">>> [Server Error] Exception during encoding '{model_name}':")
        traceback.print_exc()
        print("!" * 60 + "\n")
        err_type = type(e).__name__
        if isinstance(e, FileNotFoundError):
            clean_detail = f"Model files for '{model_name}' could not be located on the server."
        else:
            clean_detail = f"Server execution error ({err_type}) during encoding. Please inspect host console logs."
        raise HTTPException(status_code=500, detail=clean_detail)


@app.post("/v1/unload")
async def manual_unload(request: Request):
    """
    Manual eviction API. Can accept an optional JSON body: {"model_name": "qwen_image_21"}.
    If model_name is omitted or empty, unloads all resident models.
    """
    model_to_evict = None
    try:
        data = await request.json()
        raw_name = data.get("model_name")
        if raw_name:
            model_to_evict = os.path.basename(str(raw_name).strip())
    except Exception:
        pass

    if model_to_evict:
        await model_manager.evict_model(model_to_evict)
    else:
        await model_manager.unload_all()

    return {
        "status": "ok",
        "remaining_models": list(model_manager.pool.keys())
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
