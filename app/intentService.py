import os
import json
from typing import Optional, Dict, Any
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from google.cloud import storage


# config
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "mount").lower()  # mount || "gcs"
MODEL_KIND = os.getenv("MODEL_KIND", "tfidf").lower()  # tfidf or bert

# mount mode
MOUNT_DIR = os.getenv("MOUNT_DIR", "/mnt/models")
MOUNT_MODEL_FILE = os.getenv("MOUNT_MODEL_FILE", "model.joblib")
MOUNT_META_FILE = os.getenv("MOUNT_META_FILE", "metadata.json")

# gcs mode
GCS_BUCKET = os.getenv("GCS_BUCKET", "")  # name of bucket
GCS_PREFIX = os.getenv("GCS_PREFIX", "")  # folder path in bucket
GCS_KEY_PATH = os.getenv("GCS_KEY_PATH", "")  # path to gcs service account key
LOCAL_CACHE_DIR = os.getenv(
    "LOCAL_CACHE_DIR", "./.cache_model"
)  # where to cache gcs model locally

RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "")

# set title and description for swagger documentation
app = FastAPI(
    title="Intent Classification Service",
    description="Deterministic intent routing classifier.",
)

# global state for loaded model and metadata
_model = None
_metadata: Dict[str, Any] = {}
_model_path = None
_meta_path = None


# validate request shape
class ClassifyRequest(BaseModel):
    text: str


# create directory if not exists
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_from_mount() -> tuple[object, Dict[str, Any], str, str]:
    # set file paths for model and metadata
    model_path = os.path.join(MOUNT_DIR, MOUNT_MODEL_FILE)
    meta_path = os.path.join(MOUNT_DIR, MOUNT_META_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found at {model_path}")

    # load model from mount
    model = joblib.load(model_path)

    # load metadata from mount if exists
    meta: Dict[str, Any] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta, model_path, meta_path


def download_from_gcs() -> tuple[str, str]:
    # error handling for missing config
    if not GCS_BUCKET or not GCS_PREFIX:
        raise RuntimeError("missing gcs bucket or prefix")

    if not GCS_KEY_PATH or not os.path.exists(GCS_KEY_PATH):
        raise RuntimeError("missing gcs key path for local mode")

    # create directory for cache
    ensure_dir(LOCAL_CACHE_DIR)

    ## file paths for local cache
    model_path = os.path.join(LOCAL_CACHE_DIR, "model.joblib")
    meta_path = os.path.join(LOCAL_CACHE_DIR, "metadata.json")

    # login and get bucket
    client = storage.Client.from_service_account_json(GCS_KEY_PATH)
    bucket = client.bucket(GCS_BUCKET)

    model_blob = bucket.blob(f"{GCS_PREFIX}/model.joblib")
    meta_blob = bucket.blob(f"{GCS_PREFIX}/metadata.json")

    if not model_blob.exists(client=client):
        raise FileNotFoundError(
            f"gcs model not found gs://{GCS_BUCKET}/{GCS_PREFIX}/model.joblib"
        )

    # download model
    model_blob.download_to_filename(model_path)

    # download metadata if exists, otherwise create empty json
    meta_blob.download_to_filename(meta_path)

    return model_path, meta_path


def load_from_gcs() -> tuple[object, Dict[str, Any], str, str]:
    # download model and metadata from gcs to local cache
    model_path, meta_path = download_from_gcs()
    model = joblib.load(model_path)  # load model from local cache

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)  # load metadata from local cache

    return model, meta, model_path, meta_path


def load_model() -> None:
    global _model, _metadata, _model_path, _meta_path

    # check if mount or gcs
    if MODEL_SOURCE == "mount":
        model, meta, mp, metap = load_from_mount()
    elif MODEL_SOURCE == "gcs":
        model, meta, mp, metap = load_from_gcs()
    else:
        raise RuntimeError("MODEL_SOURCE must be mount or gcs")

    # update global state
    _model = model
    _metadata = meta or {}
    _model_path = mp
    _meta_path = metap


def _get_classes(model: object) -> list[str]:
    # pipeline
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        clf = model.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]

    # plain estimator
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]

    return []


def _predict_proba(
    model: object, text: str
) -> tuple[str, float, float, Dict[str, float]]:
    # predict probabilities using pipeline or plain estimator
    probs = model.predict_proba([text])[0]
    # grab classes from pipeline or plain estimator
    classes = model.named_steps["clf"].classes_

    # get top intent, confidence
    top_idx = int(np.argmax(probs))
    top_label = str(classes[top_idx])
    top_prob = float(probs[top_idx])

    # get margin between top 2 intents
    second_prob = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
    margin = top_prob - second_prob

    # map class to probability
    prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
    return top_label, top_prob, margin, prob_map


# load model on startup
@app.on_event("startup")
def _startup():
    load_model()


# health check endpoint
@app.get("/health")
def health():
    # check if model is loaded
    if _model is None:
        raise HTTPException(status_code=500, detail="model not loaded")

    # get labels for logging
    labels = _get_classes(_model)

    # return health status and model info
    return {
        "ok": True,
        "model_source": MODEL_SOURCE,
        "model_kind": MODEL_KIND,
        "model_path": _model_path,
        "labels_count": len(labels),
    }


@app.get("/model")
def model_info():
    if _model is None:
        raise HTTPException(status_code=500, detail="model not loaded")

    # info about model, including labels and metadata
    return {
        "model_source": MODEL_SOURCE,
        "model_kind": MODEL_KIND,
        "model_path": _model_path,
        "metadata_path": _meta_path,
        "labels": _get_classes(_model),
        "metadata": _metadata,
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="model not loaded")

    # clean up text
    text = (req.text or "").strip()

    # throw error if text is empty
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    # predict intent and confidence
    top_label, top_prob, margin, prob_map = _predict_proba(_model, text)

    # return predicted intent, confidence, margin, and tool state
    return {
        "top_intent": top_label,
        "confidence": top_prob,
        "margin": margin,
        "probs": prob_map,
    }


# endpoint to reload model, protected by optional token in header
@app.post("/reload")
def reload_model(x_reload_token: Optional[str] = Header(default="")):
    if RELOAD_TOKEN:
        if x_reload_token != RELOAD_TOKEN:
            raise HTTPException(status_code=403, detail="forbidden")

    # load model
    load_model()

    # ok message
    return {
        "ok": True,
        "model_path": _model_path,
        "model_source": MODEL_SOURCE,
    }
