# RadiantAI MVP Deployment Guide

This is the deployment story for the final GitHub MVP.

## Architecture

```text
GitHub repo
  -> Render hosts FastAPI backend
  -> Vercel hosts React frontend
  -> Google Gemini 2.5 Flash powers AI analysis and chat
  -> Product-grounded retrieval feeds Sephora, StyleKorean, Amazon, and Korean brand candidates into chat
  -> Local chat memory keeps user skin type, budget, concerns, allergies, and brand preferences
  -> Google Vision API is optional supporting image intelligence
  -> AWS S3 stores uploaded, metadata-scrubbed images
```

## 1. FastAPI Backend

FastAPI is the Python API server in `backend/src/main.py`.

It exposes:

- `GET /` health check
- `POST /upload` skin image analysis and recommendations
- `POST /recommend` budget/routine recalculation
- `POST /api/chat` skincare chatbot
- `POST /api/image-url` fresh image URL lookup
- `POST /api/live-products/search` product search

Local command:

```bash
uvicorn backend.src.main:app --host 127.0.0.1 --port 8000 --reload
```

Production command inside Docker:

```bash
uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 1
```

## 2. Google Gemini 2.5 Flash

Gemini powers:

- image analysis in `backend/src/services/analysis.py`
- chatbot answers in `backend/src/services/chatbot.py`

The chatbot is wrapped with extra MVP intelligence:

- product retrieval before each product/routine answer
- exact catalog products passed into Gemini as context
- user memory summary passed into Gemini as context
- backend guardrails that still return catalog product cards if Gemini names do not match exactly
- evaluation tests in `backend/tests/test_chat_intelligence.py`

Set this secret locally and on Render:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key
```

Do not commit the real key to GitHub.

## 3. Google Vision API

Google Vision is optional for this MVP. If you enable it, use a Google Cloud service account JSON file.

Local `.env`:

```env
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/google-service-account.json
```

For Render, do not commit the JSON file. Use Render secret files or another secure secret mechanism, then point `GOOGLE_APPLICATION_CREDENTIALS` at that secret file path.

If Vision is not configured, the backend can still run with Gemini.

## 4. AWS S3 Storage

S3 stores scrubbed uploaded images in production. The app now automatically uses S3 when `S3_BUCKET_NAME` is configured.

Create an S3 bucket, then set these Render environment variables:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
S3_UPLOAD_PREFIX=uploads
S3_PRESIGNED_URL_EXPIRES_SECONDS=3600
```

Recommended S3 settings:

- Keep public access blocked.
- Let the backend return presigned URLs instead of making images public.
- Use an IAM user/policy that only allows access to this one bucket.

Minimum IAM actions for the MVP:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/uploads/*"
    }
  ]
}
```

## 5. Render Backend Hosting

The repo includes `render.yaml`.

Render should deploy the backend as:

- service type: Web Service
- environment: Docker
- root directory: `backend`
- health check path: `/`

Render environment variables:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key
CORS_ORIGINS=https://your-vercel-app.vercel.app
BACKEND_PUBLIC_URL=https://your-render-backend.onrender.com
ENABLE_LIVE_PRODUCT_SEARCH=true
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
S3_UPLOAD_PREFIX=uploads
```

After deploy, test:

```text
https://your-render-backend.onrender.com/
https://your-render-backend.onrender.com/docs
```

## 6. Vercel Frontend Hosting

The frontend is a Vite React app in `frontend/`.

Deploy it on Vercel with:

- framework preset: Vite
- root directory: `frontend`
- build command: `npm run build`
- output directory: `dist`

Set this Vercel environment variable:

```env
VITE_API_URL=https://your-render-backend.onrender.com
```

After the frontend deploys, copy the Vercel URL and update Render:

```env
CORS_ORIGINS=https://your-vercel-app.vercel.app
```

Then redeploy the Render backend.

## 7. GitHub MVP Checklist

Before presenting:

- Make sure `.env` is not committed.
- Include `.env.example`.
- Include `render.yaml`.
- Include `backend/Dockerfile`.
- Include `docs/MVP_DEPLOYMENT.md`.
- Put the live frontend URL in the GitHub repo description.
- Put the backend docs URL in the README or project notes.
- Show a short demo flow: upload image -> analysis -> product routine -> chatbot follow-up -> chat history.

## Official Docs

- Google Gemini API keys: https://ai.google.dev/gemini-api/docs/api-key
- Google Vision setup: https://docs.cloud.google.com/vision/docs/setup
- Amazon S3 getting started: https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html
- Vercel environment variables: https://vercel.com/docs/environment-variables
