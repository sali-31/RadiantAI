# Deployment Guide for RadiantAI

This document outlines two deployment strategies:
1.  **Option A: Render + Vercel (Recommended for Speed)** - Easiest for React/FastAPI.
2.  **Option B: Google Cloud Platform (Enterprise)** - Full cloud-native deployment.

---

# Option A: Render + Vercel (Quick Start)

This is the recommended path for rapid deployment and prototyping.

## Prerequisites
-   GitHub Account
-   [Render](https://render.com/) Account (Backend)
-   [Vercel](https://vercel.com/) Account (Frontend)
-   AWS Account (S3) & Google Cloud Account (Gemini)

## 1. Backend Deployment (Render)

1.  **Prepare**: Ensure `backend/requirements.txt` is updated (`pip freeze > requirements.txt`).
2.  **Create Service**:
    *   Go to Render Dashboard -> New Web Service.
    *   Connect your GitHub repo.
    *   **Root Directory**: `backend`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port 10000`
3.  **Environment Variables**:
    *   Add `GOOGLE_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET_NAME`.
4.  **Deploy**: Click Create. Copy the resulting URL (e.g., `https://RadiantAI-backend.onrender.com`).

## 2. Frontend Deployment (Vercel)

1.  **Configure**: Create `frontend/.env.production` locally (optional, or set in Vercel UI).
2.  **Create Project**:
    *   Go to Vercel Dashboard -> Add New Project.
    *   Import GitHub repo.
    *   **Root Directory**: `frontend`.
    *   **Build Command**: `npm run build`.
3.  **Environment Variables**:
    *   `VITE_API_URL`: Your Render Backend URL (e.g., `https://RadiantAI-backend.onrender.com`).
4.  **Deploy**: Click Deploy.

## 3. Final Connection
1.  Copy your new Vercel domain (e.g., `https://RadiantAI.vercel.app`).
2.  Go back to Render -> Environment Variables.
3.  Add `FRONTEND_URL` = `https://RadiantAI.vercel.app`.
4.  Redeploy Render to update CORS settings.

---

# Option B: Google Cloud Platform (Enterprise)

## Overview

This guide walks through deploying the RadiantAI skin analysis system to Google Cloud Platform (GCP).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Browser                         │
│                  (React Frontend)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Cloud Run (Backend API)                     │
│            - FastAPI Application                         │
│            - Product Recommendations                     │
│            - AI Analysis Integration                     │
└────────────┬───────────────────┬────────────────────────┘
             │                   │
             ▼                   ▼
┌────────────────────┐  ┌───────────────────┐
│   Cloud Storage    │  │  Gemini AI API    │
│   (S3 → GCS)       │  │  Vision AI API    │
│   - User Images    │  │  - Analysis       │
└────────────────────┘  └───────────────────┘
```

## Prerequisites

1. **Google Cloud Account**
   - Sign up at https://cloud.google.com
   - $300 free credit for new users

2. **Google Cloud SDK (gcloud CLI)**
   ```bash
   # Install gcloud CLI
   # macOS
   brew install --cask google-cloud-sdk

   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

3. **Docker** (for containerization)
   ```bash
   # macOS
   brew install --cask docker
   ```

## Step-by-Step Setup

### 1. Create New Google Cloud Project

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (choose a unique name)
export PROJECT_ID="RadiantAI-prod"

# Create new project
gcloud projects create $PROJECT_ID --name="RadiantAI"

# Set as active project
gcloud config set project $PROJECT_ID

# Get your project number (needed for some APIs)
gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
```

### 2. Enable Required APIs

```bash
# Enable all necessary APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage-api.googleapis.com \
    storage-component.googleapis.com \
    aiplatform.googleapis.com \
    vision.googleapis.com \
    generativelanguage.googleapis.com \
    artifactregistry.googleapis.com
```

### 3. Set Up Billing

```bash
# Link billing account (required for Cloud Run)
# First, list your billing accounts
gcloud billing accounts list

# Link billing to project (replace BILLING_ACCOUNT_ID)
gcloud billing projects link $PROJECT_ID \
    --billing-account=BILLING_ACCOUNT_ID
```

### 4. Create Cloud Storage Bucket (Replacing S3)

```bash
# Set bucket name (must be globally unique)
export BUCKET_NAME="${PROJECT_ID}-uploads"

# Create bucket in us-central1
gsutil mb -l us-central1 gs://${BUCKET_NAME}

# Set lifecycle policy to delete files older than 30 days (optional)
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://${BUCKET_NAME}

# Enable CORS for web access
cat > cors.json <<EOF
[
  {
    "origin": ["http://localhost:5173", "https://your-frontend-domain.com"],
    "method": ["GET", "HEAD", "PUT", "POST", "DELETE"],
    "responseHeader": ["Content-Type"],
    "maxAgeSeconds": 3600
  }
]
EOF

gsutil cors set cors.json gs://${BUCKET_NAME}
```

### 5. Set Up Service Account

```bash
# Create service account for the application
gcloud iam service-accounts create RadiantAI-backend \
    --display-name="RadiantAI Backend Service Account"

# Grant necessary permissions
export SERVICE_ACCOUNT="RadiantAI-backend@${PROJECT_ID}.iam.gserviceaccount.com"

# Storage permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin"

# Vision AI permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/aiplatform.user"

# Cloud Run permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/run.invoker"

# Create and download service account key
gcloud iam service-accounts keys create ./service-account-key.json \
    --iam-account=$SERVICE_ACCOUNT

# IMPORTANT: Add to .gitignore
echo "service-account-key.json" >> .gitignore
```

### 6. Update Backend Configuration

Create a new configuration file for Google Cloud:

**`backend/config/gcp.py`**:
```python
import os
from pathlib import Path

# Google Cloud Project Settings
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "RadiantAI-prod")
REGION = os.getenv("GCP_REGION", "us-central1")

# Cloud Storage (replaces S3)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{PROJECT_ID}-uploads")

# Service Account
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path(__file__).parent.parent / "service-account-key.json")
)

# API Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  # Your existing Gemini key
```

### 7. Update Backend Code for GCS

Update `backend/src/main.py` to use Google Cloud Storage instead of S3:

```python
# Replace AWS S3 imports with Google Cloud Storage
from google.cloud import storage

# Initialize GCS client (replaces S3)
try:
    storage_client = storage.Client()
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    bucket = storage_client.bucket(BUCKET_NAME)
    logger.info("✓ GCS Client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GCS client: {e}")
    bucket = None

# In your upload endpoint, replace S3 upload with GCS upload:
# OLD S3 code:
# s3_client.upload_fileobj(file_obj, BUCKET_NAME, file_key)

# NEW GCS code:
blob = bucket.blob(file_key)
blob.upload_from_file(file_obj, content_type=file.content_type)
gcs_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{file_key}"
```

### 8. Create Dockerfile

**`backend/Dockerfile`**:
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy enriched product datasets
COPY backend/data/*_enriched.csv backend/data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD exec uvicorn backend.src.main:app --host 0.0.0.0 --port $PORT
```

### 9. Create .dockerignore

**`backend/.dockerignore`**:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.pytest_cache/
*.log
.git
.gitignore
.env
.DS_Store
*.md
service-account-key.json
test_*.py
```

### 10. Update Environment Variables

**`backend/.env.production`**:
```bash
# Google Cloud
GCP_PROJECT_ID=RadiantAI-prod
GCP_REGION=us-central1
GCS_BUCKET_NAME=RadiantAI-prod-uploads
GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

# API Keys
GOOGLE_API_KEY=your_gemini_api_key_here

# Application Settings
DEBUG=False
MAX_FILE_SIZE=10485760
ALLOWED_ORIGINS=https://your-frontend-domain.com

# FastAPI
PORT=8080
```

### 11. Update requirements.txt

Add Google Cloud dependencies:

```bash
# Add to backend/requirements.txt
google-cloud-storage==2.10.0
google-cloud-vision==3.4.5
google-generativeai==0.3.2
google-auth==2.23.4
```

### 12. Build and Test Locally with Docker

```bash
# Navigate to backend directory
cd backend

# Build Docker image
docker build -t RadiantAI-backend .

# Test locally
docker run -p 8080:8080 \
    -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
    -e GCP_PROJECT_ID=$PROJECT_ID \
    -e GCS_BUCKET_NAME=$BUCKET_NAME \
    -v $(pwd)/service-account-key.json:/app/service-account-key.json \
    RadiantAI-backend

# Test the API
curl http://localhost:8080/
```

### 13. Deploy to Cloud Run

```bash
# Set environment variables
export REGION="us-central1"
export SERVICE_NAME="RadiantAI-api"

# Build and deploy using Cloud Build
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --service-account=$SERVICE_ACCOUNT \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCS_BUCKET_NAME=${BUCKET_NAME}" \
    --set-secrets="GOOGLE_API_KEY=gemini-api-key:latest" \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --max-instances=10 \
    --min-instances=0

# Get the service URL
gcloud run services describe $SERVICE_NAME \
    --region=$REGION \
    --format="value(status.url)"
```

### 14. Store Secrets in Secret Manager

```bash
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create secret for Gemini API key
echo -n "your_gemini_api_key_here" | \
    gcloud secrets create gemini-api-key \
    --data-file=-

# Grant service account access to secret
gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"
```

### 15. Set Up Custom Domain (Optional)

```bash
# Map custom domain to Cloud Run service
gcloud run domain-mappings create \
    --service=$SERVICE_NAME \
    --domain=api.yourdomain.com \
    --region=$REGION

# Follow DNS verification instructions
# Add the provided DNS records to your domain registrar
```

### 16. Set Up CI/CD with Cloud Build (Optional)

**`cloudbuild.yaml`**:
```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/RadiantAI-backend:$COMMIT_SHA', './backend']

  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/RadiantAI-backend:$COMMIT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'RadiantAI-api'
      - '--image'
      - 'gcr.io/$PROJECT_ID/RadiantAI-backend:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'

images:
  - 'gcr.io/$PROJECT_ID/RadiantAI-backend:$COMMIT_SHA'
```

### 17. Update Frontend Configuration

Update your frontend to use the Cloud Run URL:

**`frontend/.env.production`**:
```bash
VITE_API_URL=https://RadiantAI-api-xxxxx-uc.a.run.app
```

### 18. Monitor and Debug

```bash
# View logs
gcloud run services logs read $SERVICE_NAME \
    --region=$REGION \
    --limit=50

# Follow logs in real-time
gcloud run services logs tail $SERVICE_NAME \
    --region=$REGION

# Check service details
gcloud run services describe $SERVICE_NAME \
    --region=$REGION

# View metrics in Cloud Console
# Navigate to: Cloud Run > RadiantAI-api > Metrics
```

## Cost Optimization

### Cloud Run Pricing (as of 2024):
- **Free tier**: 2 million requests/month
- **CPU**: $0.00002400/vCPU-second
- **Memory**: $0.00000250/GiB-second
- **Requests**: $0.40 per million requests

### Estimated Monthly Costs:
```
Scenario: 10,000 image analyses/month
- Cloud Run: ~$5-10
- Cloud Storage: ~$1-2
- Gemini API: ~$5-15 (depends on usage)
- Vision API: ~$10-20
Total: ~$21-47/month
```

### Cost Reduction Tips:
1. Set `--min-instances=0` to scale to zero when idle
2. Use `--cpu-throttling` to reduce CPU usage when idle
3. Set up budget alerts in Cloud Console
4. Use Cloud Storage lifecycle policies to delete old images

## Security Best Practices

1. **Never commit secrets**:
   ```bash
   # Add to .gitignore
   .env.production
   service-account-key.json
   *.json
   ```

2. **Use Secret Manager** for all sensitive data

3. **Enable VPC Service Controls** for production:
   ```bash
   gcloud services vpc-peerings connect \
       --service=servicenetworking.googleapis.com \
       --ranges=google-managed-services-default \
       --network=default
   ```

4. **Set up Cloud Armor** for DDoS protection

5. **Enable Cloud IAM** for fine-grained access control

## Troubleshooting

### Issue: Service won't deploy
```bash
# Check build logs
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')

# Check service logs
gcloud run services logs read $SERVICE_NAME --limit=100
```

### Issue: Can't access Cloud Storage
```bash
# Verify service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:$SERVICE_ACCOUNT"

# Test bucket access
gsutil ls gs://$BUCKET_NAME
```

### Issue: API not responding
```bash
# Check if service is running
gcloud run services list --region=$REGION

# Check service health
curl https://your-service-url.run.app/

# View recent errors
gcloud run services logs read $SERVICE_NAME \
    --region=$REGION \
    --filter="severity>=ERROR" \
    --limit=50
```

## Rollback

If something goes wrong, rollback to previous revision:

```bash
# List revisions
gcloud run revisions list \
    --service=$SERVICE_NAME \
    --region=$REGION

# Rollback to specific revision
gcloud run services update-traffic $SERVICE_NAME \
    --region=$REGION \
    --to-revisions=REVISION_NAME=100
```

## Clean Up (if needed)

```bash
# Delete Cloud Run service
gcloud run services delete $SERVICE_NAME --region=$REGION

# Delete Cloud Storage bucket
gsutil rm -r gs://$BUCKET_NAME

# Delete service account
gcloud iam service-accounts delete $SERVICE_ACCOUNT

# Delete project (WARNING: irreversible)
gcloud projects delete $PROJECT_ID
```

## Next Steps

1. ✅ Deploy backend to Cloud Run
2. ✅ Deploy frontend to Firebase Hosting or Cloud Storage + Cloud CDN
3. ✅ Set up Cloud Monitoring and Alerting
4. ✅ Configure Cloud Armor for security
5. ✅ Set up Cloud CDN for frontend assets
6. ✅ Implement Cloud Logging for analytics

## Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)

---

**Need Help?** Check Cloud Console logs or contact GCP support.
