#!/usr/bin/env python3
"""
Minimal OpenVLA inference server.

Endpoints:
  GET  /health
  POST /predict   -> multipart/form-data with fields:
                      - prompt: str
                      - image: 1 RGB image (client composes multi-cam if needed)

Notes
- Single image per request to keep processor batching simple.
- Returns: { ok, action: [floats], timings: {...} }
"""

import io
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoConfig,
    AutoImageProcessor,
)

# -----------------------------
# Configuration & local caches
# -----------------------------
MODEL_ID = os.environ.get("OPENVLA_MODEL", "openvla/openvla-7b")

# Create repo-local HF caches so we never touch /cache
HF_BASE = Path(__file__).parent / ".hf_cache"
(HF_BASE / "hub").mkdir(parents=True, exist_ok=True)
(HF_BASE / "transformers").mkdir(parents=True, exist_ok=True)
(HF_BASE / "datasets").mkdir(parents=True, exist_ok=True)

# Respect user-provided envs, otherwise default to local dirs
os.environ.setdefault("HF_HOME", str(HF_BASE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_BASE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_BASE / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_BASE / "datasets"))
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", str(HF_BASE / "transformers"))

# No gradients needed for inference
torch.set_grad_enabled(False)

app = FastAPI(title="OpenVLA Server", version="1.1")


def _cuda_bf16_supported() -> bool:
    # Compatible with older/newer torch
    try:
        fn = getattr(torch.cuda, "is_bf16_supported", None)
        return bool(fn()) if callable(fn) else False
    except Exception:
        return False


def _device_info() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    return {
        "ok": True,
        "device": str(dev),
        "attn": "sdpa" if cuda else "default",
        "bf16": _cuda_bf16_supported(),
        "model": MODEL_ID,
        "torch": torch.__version__,
    }


class Model:
    processor = None
    model = None
    device = None
    dtype = None


@app.on_event("startup")
def load_model():
    # Optional: register prismatic classes (no-op if already registered)
    try:
        from prismatic_hf.configuration_prismatic import OpenVLAConfig  # noqa: F401
        from prismatic_hf.processing_prismatic import (  # noqa: F401
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic_hf.modeling_prismatic import (  # noqa: F401
            OpenVLAForActionPrediction,
        )
        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
        # AutoProcessor & AutoModelForVision2Seq registration handled by the wheel/above import.
    except Exception:
        # If using a wheel/repo that already registers classes, this is fine.
        pass

    # Device + dtype
    Model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if Model.device.type == "cuda":
        Model.dtype = torch.bfloat16 if _cuda_bf16_supported() else torch.float16
        # Safe perf knobs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        Model.dtype = torch.float32

    # Load processor/model with explicit, writable cache_dir
    Model.processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    Model.model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=Model.dtype,
        device_map="auto" if Model.device.type == "cuda" else None,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    if Model.device.type == "cpu":
        Model.model.to(Model.device)

    # Light warm-up to reduce first-request latency
    try:
        img = Image.new("RGB", (64, 64), (0, 0, 0))
        enc = Model.processor("warmup", img, return_tensors="pt")
        for k, v in enc.items():
            if hasattr(v, "to"):
                enc[k] = v.to(Model.device, dtype=Model.dtype if v.dtype.is_floating_point else None)
        with torch.inference_mode():
            _ = Model.model.predict_action(**enc, unnorm_key="bridge_orig")
    except Exception:
        # Warmup is best-effort; ignore failures here
        pass


@app.get("/health")
def health():
    try:
        return JSONResponse(_device_info())
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"})


@app.post("/predict")
async def predict(prompt: str = Form(...), image: UploadFile = Form(...)):
    t0 = time.time()

    # Decode image
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Bad image: {e}"}, status_code=400)
    t1 = time.time()

    # Preprocess
    try:
        enc = Model.processor(prompt, img, return_tensors="pt")
        for k, v in enc.items():
            if hasattr(v, "to"):
                enc[k] = v.to(Model.device, dtype=Model.dtype if v.dtype.is_floating_point else None)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Processor error: {e}"}, status_code=500)
    t2 = time.time()

    # Inference
    try:
        with torch.inference_mode():
            action = Model.model.predict_action(**enc, unnorm_key="bridge_orig")
        # Normalize to plain Python floats
        if hasattr(action, "detach"):
            action = action.detach().float().cpu().tolist()
        elif hasattr(action, "tolist"):
            action = action.tolist()
        else:
            action = [float(x) for x in action]
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Inference error: {e}"}, status_code=500)
    t3 = time.time()

    return JSONResponse(
        {
            "ok": True,
            "action": action,
            "timings": {
                "decode_ms": round((t1 - t0) * 1000, 1),
                "preproc_ms": round((t2 - t1) * 1000, 1),
                "infer_ms": round((t3 - t2) * 1000, 1),
                "total_ms": round((t3 - t0) * 1000, 1),
            },
        }
    )
