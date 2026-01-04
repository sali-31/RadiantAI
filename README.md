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

## Privacy and Ethics

RadiantAI is designed using privacy-by-design principles:

- No image metadata is stored or processed  
- Uploaded images are used strictly for analysis  
- The system avoids storing personally identifiable information  
- Recommendations are advisory and do not constitute medical diagnoses  

---

## How to Run (Development)
# Backend
cd backend
pip install -r requirements.txt
python src/main.py

# Frontend
cd frontend
npm install
npm run dev

## One-command Development (Optional)

./start-dev.sh   # macOS/Linux
start-dev.bat    # Windows

---
## Future Work

-Quantitative evaluation of model accuracy across diverse skin tones and lighting conditions

-Bias and fairness analysis in computer vision predictions

-Causal inference for product effectiveness

-Deployment with monitored inference and feedback loops

-Expanded conversational AI capabilities

---

RadiantAI was developed as an independent, self-directed project to apply advanced concepts in data science, machine learning, and system design.

