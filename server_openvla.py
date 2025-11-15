#!/usr/bin/env python3
import io, os, time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
)

# -------------------- Model + cache setup --------------------
MODEL_ID = os.environ.get("OPENVLA_MODEL", "openvla/openvla-7b")
REVISION = os.environ.get(
    "OPENVLA_REVISION",
    "31f090d05236101ebfc381b61c674dd4746d4ce0",  # pin to avoid surprise code updates
)

BASE = Path(__file__).parent
HF_BASE = BASE / ".hf_cache"
(HF_BASE / "hub").mkdir(parents=True, exist_ok=True)
(HF_BASE / "transformers").mkdir(parents=True, exist_ok=True)
(HF_BASE / "datasets").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_BASE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_BASE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_BASE / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_BASE / "datasets"))

# Allow fully offline after first snapshot if desired: export HF_LOCAL_ONLY=1
LOCAL_ONLY = os.environ.get("HF_LOCAL_ONLY", "0") == "1"
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", str(HF_BASE / "transformers"))

torch.set_grad_enabled(False)
app = FastAPI(title="OpenVLA Server", version="1.1")


def _device_info():
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    try:
        is_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        is_bf16_supported = False
    return {
        "ok": True,
        "device": str(dev),
        "attn": "eager",  # we force eager below to dodge sdpa/flash checks
        "bf16": is_bf16_supported,
        "model": MODEL_ID,
        "torch": torch.__version__,
        "revision": REVISION,
        "local_only": LOCAL_ONLY,
    }


class Model:
    processor = None
    model = None
    device = None
    dtype = None


def _force_eager_attention():
    # Disable PyTorch SDPA paths so HF won't try to auto-pick them.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


@app.on_event("startup")
def load_model():
    # Register prismatic classes if they’re not already registered.
    try:
        from prismatic_hf.configuration_prismatic import OpenVLAConfig  # noqa
        from prismatic_hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa
        from prismatic_hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa

        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)
    except Exception:
        pass

    Model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if Model.device.type == "cuda":
        try:
            bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception:
            bf16_ok = False
        Model.dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        Model.dtype = torch.float32

    _force_eager_attention()

    # --- load processor (local if possible) ---
    Model.processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        local_files_only=LOCAL_ONLY,
        resume_download=not LOCAL_ONLY,
        use_fast=True,  # suppresses the "slow processor" warning when available
    )

    # --- load config with forced eager attention ---
    cfg = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        local_files_only=LOCAL_ONLY,
        resume_download=not LOCAL_ONLY,
    )
    # Make both the public and the internal flags explicit.
    setattr(cfg, "attn_implementation", "eager")
    setattr(cfg, "_attn_implementation_internal", "eager")

    # --- load model using that config (prevents SDPA probe) ---
    Model.model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        config=cfg,
        dtype=Model.dtype,                    # newer arg name (torch_dtype is deprecated)
        device_map="auto" if Model.device.type == "cuda" else None,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        local_files_only=LOCAL_ONLY,
        resume_download=not LOCAL_ONLY,
        low_cpu_mem_usage=True,
    )

    if Model.device.type == "cpu":
        Model.model.to(Model.device)


@app.get("/health")
def health():
    try:
        return JSONResponse(_device_info())
    except Exception as e:
        return JSONResponse({"ok": False, "error": repr(e)}, status_code=500)


@app.post("/predict")
async def predict(prompt: str = Form(...), image: UploadFile = Form(...)):
    t0 = time.time()
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Bad image: {e}"}, status_code=400)

    t1 = time.time()
    try:
        enc = Model.processor(prompt, img, return_tensors="pt")
        for k, v in enc.items():
            if hasattr(v, "to"):
                enc[k] = v.to(Model.device, dtype=Model.dtype if v.dtype.is_floating_point else None)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Processor error: {e}"}, status_code=500)

    t2 = time.time()
    try:
        action = Model.model.predict_action(**enc, unnorm_key="bridge_orig")
        if hasattr(action, "detach"):
            action = action.detach().float().cpu().tolist()
        elif hasattr(action, "tolist"):
            action = action.tolist()
        else:
            action = [float(x) for x in action]
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Inference error: {e}"}, status_code=500)

    t3 = time.time()
    return JSONResponse({
        "ok": True,
        "action": action,
        "timings": {
            "decode_ms": round((t1 - t0) * 1000, 1),
            "preproc_ms": round((t2 - t1) * 1000, 1),
            "infer_ms": round((t3 - t2) * 1000, 1),
            "total_ms": round((t3 - t0) * 1000, 1),
        }
    })
