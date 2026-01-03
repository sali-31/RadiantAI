# Setup Guide for RadiantAI

This guide will walk you through setting up the RadiantAI project, a full-stack application for skin analysis and product recommendation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Cloning the Repository](#cloning-the-repository)
3. [Backend Setup (FastAPI)](#backend-setup-fastapi)
4. [Frontend Setup (React)](#frontend-setup-react)
5. [Running the Application](#running-the-application)
6. [Environment Variables](#environment-variables)

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
- **Node.js (v16+) & npm**: [Download Node.js](https://nodejs.org/)
- **Git**: [Download Git](https://git-scm.com/)

You will also need API keys for:
- **Google Cloud Platform**:
    - **Gemini API** (Generative AI)
    - **Cloud Vision API** (Image Analysis/Privacy)
- **AWS (Amazon Web Services)**:
    - **S3** (Image Storage)

---

## Cloning the Repository

```bash
git clone https://github.com/sali-31/RadiantAI.git
cd RadiantAI
```

---

## Backend Setup (FastAPI)

The backend handles image processing, AI analysis, and product recommendations.

1.  **Navigate to the project root**:
    ```bash
    cd RadiantAI
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the root directory (or `backend/`) with the following keys:

    ```env
    # Google Cloud Credentials
    GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google-service-account.json"
    GEMINI_API_KEY="your_gemini_api_key"

    # AWS Credentials (for S3 Storage)
    AWS_ACCESS_KEY_ID="your_aws_access_key"
    AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
    AWS_REGION="us-east-1"
    S3_BUCKET_NAME="your-s3-bucket-name"
    ```

    *Note: Ensure your Google Service Account has permissions for Cloud Vision API.*

---

## Frontend Setup (React)

The frontend is a modern React application built with Vite.

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```

2.  **Install Node Modules**:
    ```bash
    npm install
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in the `frontend/` directory:

    ```env
    VITE_API_URL="http://localhost:8000"
    ```

---

## Running the Application

We provide a convenience script to start both the backend and frontend servers simultaneously.

### Option 1: Using the Startup Script (Recommended)

From the root directory:

```bash
./start-dev.sh
```

This will launch:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:5173

### Option 2: Manual Startup

**Terminal 1 (Backend):**
```bash
source .venv/bin/activate
uvicorn src.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

---

## Troubleshooting

### Common Issues

1.  **"Module not found" errors**:
    - Ensure your virtual environment is activated (`source .venv/bin/activate`).
    - Run `pip install -r requirements.txt` again.

2.  **Google API Errors**:
    - Verify `GOOGLE_APPLICATION_CREDENTIALS` points to a valid JSON key file.
    - Ensure the Cloud Vision API is enabled in your Google Cloud Console.

3.  **CORS Errors**:
    - Check that the backend `CORSMiddleware` in `src/main.py` allows requests from `http://localhost:5173`.

4.  **S3 Upload Fails**:
    - Verify your AWS credentials and ensure the S3 bucket exists and has write permissions.
