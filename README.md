# RadiantAI  
Privacy-Preserving AI for Personalized Skincare Recommendations

RadiantAI is a full-stack, data-driven web application that analyzes user-uploaded skin images to generate personalized, budget-aware skincare routines. The system integrates computer vision, machine learning, and recommendation logic while prioritizing user privacy by removing image metadata before any analysis.

This project demonstrates applied skills in data preprocessing, computer vision, machine learning model integration, AI-assisted decision systems, and end-to-end system design.

---

## Problem Motivation

Skincare recommendations are often generic, expensive, or lack transparency. At the same time, users are increasingly concerned about privacy when uploading personal images.

RadiantAI addresses these challenges by:
- Analyzing skin conditions directly from images using AI
- Removing all image metadata (EXIF) to protect user privacy
- Recommending affordable, structured skincare routines
- Allowing follow-up interaction through an AI chatbot

---

## Key Features

- **Privacy-First Image Processing**
  - Automatic stripping of EXIF and embedded metadata before inference
- **AI-Driven Skin Analysis**
  - YOLO and YOLOv10-based computer vision models for acne and skin condition detection
  - Integration with Google AI services for enhanced image understanding
- **Budget-Aware Recommendation Engine**
  - Product selection across cleanser, treatment, and moisturizer categories
  - Cost-aware filtering and ranking logic
- **Interactive Dashboard**
  - Visual presentation of AI analysis and recommended routines
- **AI Chatbot**
  - Follow-up questions about product usage and skincare routines
- **End-to-End System Design**
  - Separate frontend, backend, and machine learning pipelines

---

## Technical Architecture

### Frontend
- Framework: React with TypeScript (Vite)
- Styling: Tailwind CSS
- Core functionality:
  - Secure image upload
  - Interactive dashboard for results
  - Embedded chatbot interface
- Deployment-ready configuration using Vercel

### Backend
- Language: Python
- Core responsibilities:
  - Image preprocessing and metadata removal
  - AI inference pipeline
  - Product recommendation logic
  - Chatbot service
- Containerization:
  - Docker support for reproducible environments

### Machine Learning & Data
- Computer Vision Models:
  - YOLO and YOLOv10 architectures for skin condition detection
- Training and Inference:
  - Custom training scripts
  - YAML-based model configuration files
- Data Versioning:
  - DVC (Data Version Control) for reproducible ML workflows
- Experimentation:
  - Jupyter notebooks for exploratory analysis and evaluation

---

## Data Science Focus

This project emphasizes several core Data Science competencies:

- **Data preprocessing:** image normalization, metadata removal, and structured product data cleaning  
- **Model selection and evaluation:** comparing YOLO and YOLOv10 for skin condition detection  
- **Pipeline design:** separating preprocessing, inference, and recommendation layers  
- **Reproducibility:** versioned datasets and models using DVC  
- **Human-centered AI:** interpretable outputs and user-focused recommendations  

---

##  and Ethics

RadiantAI is designed using privacy-by-design principles:

- No image metadata is stored or processed  
- Uploaded images are used strictly for analysis  
- The system avoids storing personally identifiable information  
- Recommendations are advisory and do not constitute medical diagnoses  

---

## 📂 Project Structure

### **Root Directory**
- `backend/`: Python FastAPI server and logic.
- `frontend/`: React application.
- `archive/`: Legacy code (Streamlit, YOLO models) and documentation.
- `docs/`: Current project documentation.
- `start-dev.sh`: Script to launch both frontend and backend.

### **Backend Breakdown (`backend/src/`)**
- **`main.py`**: The entry point for the FastAPI server. Defines endpoints `/upload`, `/recommend`, and `/api/chat`.
- **`services/`**:
  - `analysis.py`: Handles interaction with Google Gemini API.
  - `privacy.py`: Utilities for metadata scrubbing.
  - `product_recommender.py`: The core logic engine. Contains the "Knapsack-style" algorithm for bundling and filtering logic.
  - `chatbot.py`: Manages the conversational AI logic.
  - `product_data_cleaner.py`: Utilities for cleaning and loading CSV data.
- **`data/`**: Contains the CSV files for different skin conditions (e.g., `acne_products.csv`, `rosacea_products.csv`).

### **Frontend Breakdown (`frontend/src/`)**
- **`App.tsx`**: Main application controller. Handles routing between Dashboard, Upload, Results, and Chat views.
- **`components/`**:
  - `ImageUpload.tsx`: Manages file selection, camera streaming, and API upload calls.
  - `RecommendedProducts.tsx`: The results dashboard. Manages the budget state, sorting, pagination, and displays the bundle/list.
  - `ProductRoutine.tsx`: Renders the "Bundle" as a visual step-by-step card grid.
  - `Chatbot.tsx`: The chat interface component.
  - `ErrorBoundary.tsx`: Catches runtime errors to display a friendly fallback UI.

---

## 🔄 Data Flow Guide

1.  **User Action**: User uploads an image or captures a photo via `ImageUpload.tsx`.
2.  **Frontend**: Sends `POST /upload` request with the image file to the Backend.
3.  **Backend (Privacy)**: `privacy.py` strips any remaining metadata.
4.  **Backend (Storage)**: Uploads the processed image to AWS S3.
5.  **Backend (AI Analysis)**: `analysis.py` sends the scrubbed image to Gemini 2.0 Flash.
    - *Prompt*: "Analyze this skin image for conditions..."
    - *Response*: JSON containing `condition` (e.g., "acne"), `severity`, and `characterization`.
6.  **Backend (Recommendation)**: `product_recommender.py` takes the analysis:
    - Loads the relevant product CSV (e.g., `acne_products.csv`).
    - **Bundle Logic**: Selects a Cleanser, Treatment, and Moisturizer such that `Sum(Prices) <= Budget`.
    - **List Logic**: Selects top-rated items where `Item_Price <= Budget`.
7.  **Response**: Backend returns the Analysis + Bundle + Recommendations to Frontend.
8.  **Frontend**: `App.tsx` saves data to `localStorage` and switches to `RecommendedProducts.tsx` view.
9.  **User Interaction**: User enters a new budget (e.g., ).
10. **Update**: Frontend calls `POST /recommend` with the *existing* analysis text and *new* budget. Backend recalculates and returns the new bundle.

---

## 🧩 Planned Feature: Interactive Weekly Routines & Community Reviews

### Overview
This feature will allow users to turn their recommended product routines into shareable, customizable weekly templates. Users can:
- Create a weekly routine based on their AI-recommended bundle.
- Add a personal description/breakdown for each routine.
- Upload progress photos (e.g., 1 week, 1 month after starting routine).
- Share routines for others to review, rate, and comment.
- Copy routines from other users and substitute products as needed.

### 1. Backend Architecture Changes
**Current State**: Routines are generated on-the-fly and stored in client-side `localStorage`.
**Required Change**: Introduce a persistent database (PostgreSQL recommended) to store User Generated Content (UGC).

#### Database Schema (Draft)
*   **`Routines` Table**:
    *   `id`: UUID (Primary Key)
    *   `user_id`: String (Owner)
    *   `title`: String (e.g., "My Acne Fighting Journey")
    *   `description`: Text (User's breakdown/explanation)
    *   `products`: JSONB (List of products with `id`, `name`, `category`, `image_url`)
    *   `schedule`: JSONB (e.g., `{"Monday": ["cleanser", "treatment"], ...}`)
    *   `condition_tags`: Array (e.g., ["Acne", "Oily Skin"])
    *   `is_public`: Boolean
    *   `created_at`: Timestamp

*   **`RoutineReviews` Table**:
    *   `id`: UUID
    *   `routine_id`: UUID (Foreign Key)
    *   `reviewer_id`: String
    *   `rating`: Integer (1-5)
    *   `comment`: Text
    *   `created_at`: Timestamp

*   **`ProgressPhotos` Table**:
    *   `id`: UUID
    *   `routine_id`: UUID
    *   `s3_key`: String (Path to image in S3)
    *   `stage`: String (e.g., "Week 1", "Month 1")
    *   `caption`: Text

### 2. API Endpoints (New)
*   **Routine Management**:
    *   `POST /routines`: Save a current recommendation as a new Routine.
    *   `GET /routines`: Search/List public routines (filter by condition, rating).
    *   `GET /routines/{id}`: Get full details of a specific routine.
    *   `PUT /routines/{id}`: Update description, schedule, or products.
    *   `POST /routines/{id}/fork`: Copy another user's routine to your library (allows substitution).

*   **Community Interaction**:
    *   `POST /routines/{id}/reviews`: Submit a rating and comment.
    *   `POST /routines/{id}/photos`: Upload a progress transition photo.

### 3. Frontend Implementation Plan
*   **"Save as Routine" Action**:
    *   Add button to `RecommendedProducts.tsx` and `Dashboard.tsx`.
    *   Opens a "Create Routine" wizard to add a Title and Description.

*   **Routine Editor**:
    *   Interface to drag-and-drop products into a weekly schedule.
    *   "Substitute Product" feature: Click a product -> Search Catalog -> Replace.

*   **Community Hub**:
    *   New main navigation tab: "Community".
    *   Feed of top-rated routines and success stories (Progress Photos).

*   **Routine Detail View**:
    *   Shows the "Before/After" photos.
    *   Displays the product list with "Buy Now" links.
    *   Comments section for user reviews.

---

## 🛠️ Getting Started

### Prerequisites
- Node.js (v16+)
- Python (v3.9+)
- Google Cloud Credentials (for Vision & Gemini)
- AWS Credentials (for S3)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sali-31/RadiantAI.git
    cd RadiantAI
    ```

2.  **Environment Setup**:
    Create a `.env` file in the root (or `backend/`) with the following:
    ```env
    GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google-creds.json"
    GEMINI_API_KEY="your_gemini_key"
    AWS_ACCESS_KEY_ID="your_aws_key"
    AWS_SECRET_ACCESS_KEY="your_aws_secret"
    AWS_REGION="us-east-1"
    S3_BUCKET_NAME="your-bucket-name"
    ```

3.  **Run the Application**:
    I have provided a convenience script to start both servers:
    ```bash
    ./start-dev.sh
    ```
    *This will start the Backend on `http://localhost:8000` and the Frontend on `http://localhost:5173`.*

---

## Future Work

-Quantitative evaluation of model accuracy across diverse skin tones and lighting conditions

-Bias and fairness analysis in computer vision predictions

-Causal inference for product effectiveness

-Deployment with monitored inference and feedback loops

-Expanded conversational AI capabilities

---

RadiantAI was developed as an independent, self-directed project to apply advanced concepts in data science, machine learning, and system design.


