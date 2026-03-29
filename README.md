# Intent-Classification-Service

Intent Classification Service - a minimal FastAPI microservice that loads a trained intent classifier (TF-IDF baseline, later BERT) and returns the top intent plus confidence scores for a given text input.

## local setup (gcs model source)
### 0 Clone repository

```bash
git clone <url-to-repo>
```

### 1 create and activate a virtual environment

```bash
cd Intent-Classification-Service

python3 -m venv venv
source venv/bin/activate
```

### 2 install dependencies

```bash
pip install -r requirements.txt
```

### 3 configure environment variables

set these in your terminal session

```bash
export MODEL_SOURCE="gcs"
export MODEL_KIND="tfidf"
export GCS_BUCKET="financal-advisor-bot-models"
export GCS_PREFIX="TF-IDF_models_current"
export GCS_KEY_PATH=</absolute/path/to/service-account-key.json> # service account key is provided in submission
```

note - do not commit any json keys or env files  
note - if you use a .env file, keep it gitignored

### 4 run the api locally

```bash
uvicorn app.intentService:app --host 127.0.0.1 --port 8000 --reload
```

### 5 quick test

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"what is an asset","tool_state":"OPEN_LEARNING"}'
```

## cloud run setup (mounted model source)

These environment variables are configured in the Cloud Run service configuration:

```bash
MODEL_SOURCE=mount
MODEL_KIND=tfidf
MOUNT_DIR=/mnt/models
MOUNT_MODEL_FILE=model.joblib
MOUNT_META_FILE=metadata.json
```

The service loads the model from the mounted path at startup.
These variables are configured in the Cloud Run deployment and do not need to be set locally.
