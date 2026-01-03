# API Setup Guide - Gemini & Cloud Vision

## Option A: Gemini API (Recommended - Easiest & Free)

### Get Gemini API Key:

1. **Go to Google AI Studio**:
   - Visit: https://aistudio.google.com/app/apikey
   - Sign in with your Google account

2. **Create API Key**:
   - Click "Get API key"
   - Click "Create API key in new project" (or select existing project)
   - Copy the API key (starts with `AIza...`)

3. **Add to your `.env` file**:
   ```bash
   # In /backend/.env
   GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

4. **Test it works**:
   ```bash
   cd backend
   python -c "
   import os
   from dotenv import load_dotenv
   import google.generativeai as genai

   load_dotenv()
   genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
   model = genai.GenerativeModel('gemini-1.5-flash')
   response = model.generate_content('Hello')
   print('✓ Gemini API working!')
   print(response.text)
   "
   ```

**Pricing:** FREE for up to 15 requests/minute

---

## Option B: Google Cloud Vision API (More Complex)

### Prerequisites:
- Google Cloud account (free $300 credit)
- Credit card (required for verification, won't be charged in free tier)

### Step 1: Create Google Cloud Project

```bash
# Install gcloud CLI first (if not installed)
# macOS:
brew install --cask google-cloud-sdk

# Login
gcloud auth login

# Create project
gcloud projects create RadiantAI-adon-vision --name="RadiantAI Vision"

# Set as active project
gcloud config set project RadiantAI-adon-vision

# Link billing account (REQUIRED - but free tier available)
# First, list billing accounts
gcloud billing accounts list

# Link to project (replace BILLING_ACCOUNT_ID)
gcloud billing projects link RadiantAI-adon-vision \
    --billing-account=BILLING_ACCOUNT_ID
```

### Step 2: Enable Vision API

```bash
# Enable the Vision API
gcloud services enable vision.googleapis.com
```

### Step 3: Create Service Account & Credentials

```bash
# Create service account
gcloud iam service-accounts create RadiantAI-adon-vision \
    --display-name="RadiantAI Vision Service Account"

# Grant Vision API permissions
gcloud projects add-iam-policy-binding RadiantAI-adon-vision \
    --member="serviceAccount:RadiantAI-adon-vision@RadiantAI-adon-vision.iam.gserviceaccount.com" \
    --role="roles/cloudvision.admin"

# Create and download credentials JSON
gcloud iam service-accounts keys create ./backend/vision-credentials.json \
    --iam-account=RadiantAI-adon-vision@RadiantAI-adon-vision.iam.gserviceaccount.com

# IMPORTANT: Add to .gitignore
echo "backend/vision-credentials.json" >> .gitignore
```

### Step 4: Update `.env` file

```bash
# In /backend/.env
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/backend/vision-credentials.json

# Or use relative path
GOOGLE_APPLICATION_CREDENTIALS=./backend/vision-credentials.json
```

### Step 5: Test Vision API

```bash
cd backend
python -c "
from google.cloud import vision
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './vision-credentials.json'
client = vision.ImageAnnotatorClient()

print('✓ Google Cloud Vision API working!')
print(f'Client initialized: {client}')
"
```

**Pricing:**
- FREE for first 1,000 images/month
- $1.50 per 1,000 images after that

---

## Recommended Setup (Using Both)

Since you already have the code set up for both, here's the complete `.env`:

```bash
# /backend/.env

# Gemini API (for analysis)
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Google Cloud Vision API (for object detection)
GOOGLE_APPLICATION_CREDENTIALS=./backend/vision-credentials.json

# AWS S3 (for image storage)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name

# Application Settings
DEBUG=True
MAX_FILE_SIZE=10485760
PORT=8000
```

---

## Quick Start (Gemini Only - Fastest)

If you just want to get started quickly and don't need Vision API right now:

1. **Get Gemini API key** (takes 30 seconds):
   - https://aistudio.google.com/app/apikey

2. **Update `.env`**:
   ```bash
   GOOGLE_API_KEY=your_key_here
   ```

3. **Temporarily disable Vision API** in your code:

   Edit `backend/src/services/analysis.py`:
   ```python
   # Line 16-20, change:
   try:
       vision_client = vision.ImageAnnotatorClient()
   except Exception as e:
       print(f"Vision API not configured (optional): {e}")
       vision_client = None  # This is fine - Gemini will still work
   ```

4. **Run your backend**:
   ```bash
   cd backend
   uvicorn src.main:app --reload
   ```

Your app will work with just Gemini (analysis will work, just no object detection bounding boxes).

---

## Troubleshooting

### Error: "API key not valid"
- Check that you copied the full key (starts with `AIza`)
- Make sure no extra spaces in `.env` file
- Restart your server after updating `.env`

### Error: "Could not automatically determine credentials"
- Check `GOOGLE_APPLICATION_CREDENTIALS` path is correct
- Use absolute path if relative path doesn't work
- Make sure JSON file exists and is readable

### Error: "Vision API has not been used in project"
- Run: `gcloud services enable vision.googleapis.com`
- Wait 1-2 minutes for API to activate
- Try again

---

## Which Should You Use?

| Feature | Gemini API | Cloud Vision API |
|---------|-----------|------------------|
| Setup Time | 30 seconds | 5-10 minutes |
| Cost | Free (15 req/min) | Free (1k/month) |
| Credentials | Just API key | Service account JSON |
| Analysis | ✅ Text analysis | ❌ |
| Object Detection | ❌ | ✅ Bounding boxes |
| Required | ✅ YES | Optional |

**Recommendation:**
1. Start with **Gemini only** (quick setup)
2. Add **Vision API later** if you need object detection

---

## Next Steps

Once you have the APIs working:

1. ✅ Test the `/upload` endpoint
2. ✅ Verify AI analysis returns results
3. ✅ Check product recommendations work
4. ✅ Test with sample images

Need help with any step? Let me know!
