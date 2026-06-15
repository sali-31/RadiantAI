# RadiantAI Backend Setup

This backend uses FastAPI with Google's Gemini API. Google Vision and AWS S3 are optional for the current local app.

## What You Need

- Python 3.11 or newer
- A Google AI Studio API key for Gemini
- Optional: a Google Cloud Vision service account JSON file
- Optional: AWS S3 credentials if you later wire uploads to S3

## Local Setup

1. Open Terminal.

2. Go to the project folder:

   ```bash
   cd /Users/ahmedhassan/Documents/GitHub/RadiantAI
   ```

3. Create a Python virtual environment:

   ```bash
   python3 -m venv .venv
   ```

4. Turn the virtual environment on:

   ```bash
   source .venv/bin/activate
   ```

5. Install backend dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

6. Create your private environment file:

   ```bash
   cp .env.example .env
   ```

7. Open `.env` and fill in:

   ```env
   GOOGLE_API_KEY=your_google_ai_studio_api_key_here
   ```

   Keep `.env` private. It is already ignored by Git.

8. Start the backend:

   ```bash
   uvicorn backend.src.main:app --host 127.0.0.1 --port 8000 --reload
   ```

9. Open the API docs:

   ```text
   http://127.0.0.1:8000/docs
   ```

## Quick Health Check

In a second Terminal window, run:

```bash
curl http://127.0.0.1:8000/
```

You should see:

```json
{"status":"ok","service":"RadiantAI API"}
```

## Optional Google Vision Setup

Gemini works with only `GOOGLE_API_KEY`. Google Vision needs a separate Google Cloud service account JSON file.

1. Create or choose a Google Cloud project.
2. Enable the Cloud Vision API.
3. Create a service account.
4. Download its JSON key file.
5. Add this to `.env`:

   ```env
   GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your-google-service-account.json
   ```

If you skip this, the app can still run. You may see a startup warning that Vision credentials were not found.

## Optional AWS S3 Setup

The current upload endpoint saves scrubbed images locally in `backend/uploads`. S3 credentials are only needed if you add S3 upload code later.

When you are ready for S3, add these to `.env`:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_s3_bucket_name
```

## Render Deployment Notes

This repo includes `render.yaml` for the backend.

On Render, set these environment variables:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
CORS_ORIGINS=https://your-frontend-domain.com
ENABLE_LIVE_PRODUCT_SEARCH=true
```

If you use Google Vision on Render, also add `GOOGLE_APPLICATION_CREDENTIALS` and provide the service account JSON securely using Render's secret file support or an equivalent secure method.
