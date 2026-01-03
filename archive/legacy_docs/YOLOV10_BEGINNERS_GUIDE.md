# Complete Beginner's Guide: Acne Detection with YOLOv10

## 📚 Table of Contents

1. [Introduction - What I am Building](#introduction)
2. [Prerequisites](#prerequisites)
3. [Understanding YOLOv10](#understanding-yolov10)
4. [Installation & Setup](#installation-setup)
5. [Preparing Your Dataset](#preparing-dataset)
6. [Training Your Model](#training-model)
7. [Testing & Inference](#testing-inference)
8. [Understanding Results](#understanding-results)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

---

## 1. Introduction - What I am Building {#introduction}

### 🎯 **Project Goal**

We're building an **AI-powered acne detection system** using YOLOv10 that can:
1. **Detect acne lesions** in facial images
2. **Classify** different types of acne (comedones, papules, pustules, nodules)
3. **Count lesions** and assess severity (mild, moderate, severe)
4. **Work offline** without API costs
5. **Run in real-time** on images and videos

### 🚀 **Why YOLOv10?**

**YOLOv10** is the latest (2024) YOLO model with significant improvements:

| Feature | YOLOv8 | YOLOv10 | Benefit for Acne Detection |
|---------|--------|---------|---------------------------|
| **Speed** | 100 FPS | 120-140 FPS | ⚡ 20-30% faster |
| **Small objects** | Good | Better | 🎯 Detects tiny acne lesions |
| **NMS (post-processing)** | Required | Not needed | 🧹 Cleaner predictions |
| **Clustered objects** | Good | Excellent | 👥 Better for multiple lesions |
| **Architecture** | Standard | Dual assignments | 🧠 Smarter learning |

**Bottom line**: YOLOv10 is faster, more accurate, and better for medical imaging!

### 🔍 **Real-World Example**

**Input**: Photo of a person's face
**Output**:
- Bounding boxes around each acne lesion
- Classification (comedone, papule, pustule, nodule)
- Count: "15 lesions detected"
- Severity: "Moderate acne"
- Recommendation: "Consider dermatologist consultation"

---

## 2. Prerequisites {#prerequisites}

### ✅ **What You Need to Know**

- **Basic Python** (variables, functions, loops)
- **Command line basics** (cd, ls, running Python scripts)
- **Basic ML concepts** (optional but helpful)

### 💻 **Hardware Requirements**

**Minimum**:
- CPU: Any modern processor
- RAM: 8GB
- Storage: 10GB free space
- GPU: None (can train on CPU, just slower)

**Recommended** (for faster training):
- CPU: Intel i5/i7 or AMD Ryzen 5/7
- RAM: 16GB+
- Storage: 20GB+ SSD
- GPU: NVIDIA GPU with 6GB+ VRAM (e.g., GTX 1660, RTX 3060)

**Cloud Alternative** (Free GPU):
- Google Colab (free GPU for 12 hours)
- Kaggle Notebooks (free GPU for 30 hours/week)

### 📦 **Software Requirements**

- **Python 3.8+** (3.9 or 3.10 recommended)
- **pip** (Python package manager)
- **Git** (for cloning repository)

---

## 3. Understanding YOLOv10 {#understanding-yolov10}

### 🧠 **What is YOLO?**

**YOLO** = "You Only Look Once"

Traditional object detection:
1. Scan image with sliding window
2. For each window, classify: "Is this acne?"
3. Repeat thousands of times → **SLOW** ❌

YOLO approach:
1. Look at entire image once
2. Predict all objects simultaneously → **FAST** ✅

### 🎨 **How YOLOv10 Works (Simplified)**

```
Input Image (640x640)
       ↓
[YOLOv10 Neural Network]
       ↓
Output Grid (predictions)
       ↓
Bounding Boxes + Classes + Confidence
```

**Step-by-step**:
1. **Input**: Feed image into network
2. **Feature Extraction**: Network learns patterns (red bumps, texture, etc.)
3. **Detection Head**: Predicts bounding boxes and classes
4. **Output**: List of detections with confidence scores

### 🆕 **YOLOv10 Key Innovations**

1. **NMS-Free Detection**:
   - Old way: Predict multiple overlapping boxes, then filter → slow
   - YOLOv10: Predict clean boxes directly → fast ⚡

2. **Dual Label Assignment**:
   - Trains with two strategies simultaneously
   - Better learning, especially for small objects like acne

3. **Improved Backbone**:
   - Better feature extraction for medical images
   - More efficient architecture

### 📊 **Model Sizes**

YOLOv10 comes in 6 sizes:

| Model | Params | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| **YOLOv10n** | 2.3M | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Mobile apps, prototyping |
| **YOLOv10s** | 7.2M | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced, general use |
| **YOLOv10m** | 15.4M | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **Production (recommended)** |
| **YOLOv10b** | 19.1M | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | YOLOv10-specific variant |
| **YOLOv10l** | 24.4M | ⚡⚡ | ⭐⭐⭐⭐⭐ | Research, clinical |
| **YOLOv10x** | 29.5M | ⚡ | ⭐⭐⭐⭐⭐⭐ | Maximum accuracy |

**Recommendation for beginners**: Start with **YOLOv10n** (fast experiments), then upgrade to **YOLOv10m** for production.

---

## 4. Installation & Setup {#installation-setup}

### 📥 **Step 1: Clone the Repository**

```bash
# Open terminal and run:
git clone https://github.com/YOUR_USERNAME/LesionRec.git
cd LesionRec
```

### 🐍 **Step 2: Create Virtual Environment**

**Why?** Keeps dependencies isolated and prevents conflicts.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### 📦 **Step 3: Install Dependencies**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install YOLOv10 specifically
pip install git+https://github.com/THU-MIG/yolov10.git
```

**This installs**:
- PyTorch (deep learning framework)
- Ultralytics (YOLO library)
- OpenCV (image processing)
- YOLOv10 (latest YOLO version)
- And more...

### ✅ **Step 4: Verify Installation**

```bash
# Test if everything works
python -c "from ultralytics import YOLOv10; print('✓ YOLOv10 installed!')"

# Check if GPU is available (optional)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

**Expected output**:
```
✓ YOLOv10 installed!
GPU available: True  # or False if no GPU
```

### 🔧 **Step 5: Set Up DVC (Data Version Control)**

DVC manages large image datasets efficiently.

```bash
# Add DVC to PATH (macOS)
export PATH="/Users/YOUR_USERNAME/Library/Python/3.9/bin:$PATH"

# Initialize DVC (already done in this repo)
# dvc init  # Skip this if .dvc folder exists

# Pull data from remote storage
dvc pull
```

**What this does**: Downloads acne image datasets from Google Drive.

---

## 5. Preparing Your Dataset {#preparing-dataset}

### 📂 **Dataset Structure**

For YOLOv10, we need this structure:

```
data/yolo_dataset/
├── images/
│   ├── train/        # Training images
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/          # Validation images
│   │   └── ...
│   └── test/         # Test images (optional)
│       └── ...
└── labels/
    ├── train/        # Training labels
    │   ├── img1.txt  # Same name as image
    │   ├── img2.txt
    │   └── ...
    ├── val/          # Validation labels
    │   └── ...
    └── test/         # Test labels (optional)
        └── ...
```

### 🏷️ **Label Format (YOLO format)**

Each `.txt` file contains bounding box annotations:

```
class_id x_center y_center width height
```

**Example** (`img1.txt`):
```
0 0.5 0.3 0.1 0.15
1 0.7 0.6 0.08 0.12
2 0.2 0.4 0.05 0.08
```

**Explanation**:
- `0` = class ID (0=comedone, 1=papule, 2=pustule, 3=nodule)
- `0.5` = x-center (50% from left)
- `0.3` = y-center (30% from top)
- `0.1` = width (10% of image width)
- `0.15` = height (15% of image height)

**Important**: All values are normalized (0-1 range).

### 🛠️ **Option 1: Use Existing Dataset**

If you already have acne images:

```bash
# Run the preparation script
python scripts/prepare_yolo_dataset.py
```

This script:
1. Reads raw acne images from `data/raw/`
2. Generates labels using pre-trained models
3. Splits data into train/val/test
4. Creates YOLO-format dataset

### 🖼️ **Option 2: Label Your Own Images**

Use a labeling tool to manually annotate images:

**Recommended tools**:
1. **LabelImg** (easy, offline)
2. **Roboflow** (online, auto-labeling)
3. **CVAT** (advanced, collaborative)

**LabelImg quick start**:
```bash
pip install labelImg
labelImg  # Opens GUI

# 1. Open directory with images
# 2. Draw boxes around acne lesions
# 3. Select class (comedone/papule/pustule/nodule)
# 4. Save (creates .txt files automatically)
```

### 📊 **Dataset Quality Tips**

**Good dataset checklist**:
- ✅ At least 500+ images (more is better)
- ✅ Diverse skin tones (fair, medium, dark)
- ✅ Various lighting conditions
- ✅ Different acne severities (mild, moderate, severe)
- ✅ Balanced classes (similar number of each type)
- ✅ High-quality images (not blurry)

**Data split**:
- Training: 70% (model learns from this)
- Validation: 15% (tunes hyperparameters)
- Test: 15% (final evaluation)

---

## 6. Training Your Model {#training-model}

### 🎯 **Quick Start Training**

**Simplest command** (uses all defaults):
```bash
python scripts/train_yolov10.py
```

This trains:
- Model: YOLOv10n (nano, fastest)
- Epochs: 100
- Batch size: 16
- Image size: 640x640

### ⚙️ **Custom Training**

**Train a larger model** (better accuracy):
```bash
python scripts/train_yolov10.py --model yolov10m.pt --epochs 150
```

**Train with larger batch size** (if you have good GPU):
```bash
python scripts/train_yolov10.py --batch 32
```

**Train on CPU** (if no GPU):
```bash
python scripts/train_yolov10.py --device cpu
```

**Cache images in RAM** (faster training, needs more RAM):
```bash
python scripts/train_yolov10.py --cache
```

### 📋 **All Training Options**

```bash
python scripts/train_yolov10.py \
  --model yolov10m.pt \      # Model size (n/s/m/b/l/x)
  --data config/yolov10_acne.yaml \  # Dataset config
  --epochs 150 \              # Number of epochs
  --batch 16 \                # Batch size
  --imgsz 640 \               # Image size
  --patience 20 \             # Early stopping
  --device 0 \                # GPU device (0, 1, cpu)
  --project runs/detect \     # Output directory
  --name acne_yolov10 \       # Experiment name
  --workers 8 \               # Dataloader workers
  --cache                     # Cache images in RAM
```

### 📊 **Understanding Training Output**

While training, you'll see:

```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/150     2.5G      1.234      0.856      1.432        45       640
2/150     2.5G      1.102      0.743      1.298        52       640
...
```

**What it means**:
- `Epoch`: Current training iteration (1-150)
- `GPU_mem`: GPU memory used
- `box_loss`: How wrong bounding boxes are (lower is better)
- `cls_loss`: How wrong classifications are (lower is better)
- `dfl_loss`: Distribution focal loss (lower is better)
- `Instances`: Number of objects in this batch
- `Size`: Image size

**Good training signs**:
- ✅ Losses decrease over time
- ✅ GPU memory stable (not increasing)
- ✅ Training doesn't crash

**Bad training signs**:
- ❌ Losses increase or stay flat
- ❌ GPU memory keeps growing → reduce batch size
- ❌ NaN losses → reduce learning rate

### ⏱️ **How Long Will Training Take?**

**Estimates** (100 epochs, 1000 images):
- **GPU (RTX 3060)**: 2-3 hours
- **GPU (GTX 1660)**: 4-6 hours
- **CPU**: 20-30 hours (not recommended)

**Tips for faster training**:
1. Use smaller model (yolov10n instead of yolov10x)
2. Reduce image size (--imgsz 416 instead of 640)
3. Use --cache to load images into RAM
4. Use Google Colab (free GPU)

### 🛑 **Stopping and Resuming Training**

**Stop training**: Press `Ctrl+C`

**Resume training**:
```bash
python scripts/train_yolov10.py --resume runs/detect/acne_yolov10/weights/last.pt
```

### 💾 **Where Are My Models Saved?**

After training:
```
runs/detect/acne_yolov10/
├── weights/
│   ├── best.pt    # Best model (use this!)
│   └── last.pt    # Last epoch (for resuming)
├── results.png    # Training curves
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── val_batch0_pred.jpg  # Validation predictions
```

**Most important**: `weights/best.pt` - This is your trained model!

---

## 7. Testing & Inference {#testing-inference}

### 🎯 **Quick Test**

**Test on a single image**:
```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/acne_sample_1.jpg
```

**Output**:
- Annotated image with bounding boxes
- Console output with detections
- Saved to `runs/detect/predict/`

### 📁 **Test on Multiple Images**

**Process entire directory**:
```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/
```

### 🎥 **Test on Video**

```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source video.mp4
```

### 📷 **Test on Webcam**

```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source 0  # 0 = default webcam
```

### ⚙️ **Advanced Inference Options**

**Adjust confidence threshold** (default 0.4):
```bash
python scripts/yolov10_inference.py \
  --model best.pt \
  --source image.jpg \
  --conf 0.6  # Only show detections with >60% confidence
```

**Save results as JSON**:
```bash
python scripts/yolov10_inference.py \
  --model best.pt \
  --source image.jpg \
  --save-json results.json
```

**Show results in real-time**:
```bash
python scripts/yolov10_inference.py \
  --model best.pt \
  --source image.jpg \
  --show  # Opens window with results
```

**Save cropped detections**:
```bash
python scripts/yolov10_inference.py \
  --model best.pt \
  --source image.jpg \
  --save-crop  # Saves individual acne lesion crops
```

---

## 8. Understanding Results {#understanding-results}

### 📊 **Validation Metrics Explained**

After training, you'll see metrics like:

```
mAP@50: 0.8234
mAP@50-95: 0.6543
Precision: 0.8721
Recall: 0.7854
```

**What do they mean?**

#### **1. mAP@50 (Mean Average Precision at IoU=0.5)**

**Simple explanation**: "How good is the model overall?"

- **Range**: 0.0 to 1.0 (higher is better)
- **Good value**: >0.70
- **Excellent value**: >0.85

**Interpretation**:
- `0.82` = Model correctly detects 82% of acne lesions with good box placement

#### **2. mAP@50-95**

**Simple explanation**: "How good is the model with stricter requirements?"

- Averages mAP across different IoU thresholds (0.5 to 0.95)
- Harder metric (usually lower than mAP@50)
- **Good value**: >0.50

#### **3. Precision**

**Simple explanation**: "Of all detections, how many are correct?"

- Formula: `TP / (TP + FP)`
- **High precision** = Few false positives (doesn't see acne where there isn't any)

**Example**:
- Model detects 100 lesions
- 87 are actually acne
- 13 are false alarms
- Precision = 87/100 = 0.87

#### **4. Recall**

**Simple explanation**: "Of all actual acne lesions, how many did we find?"

- Formula: `TP / (TP + FN)`
- **High recall** = Few false negatives (finds most acne lesions)

**Example**:
- Image has 100 acne lesions
- Model finds 78 of them
- Misses 22
- Recall = 78/100 = 0.78

#### **Trade-off**: Precision vs Recall

**For medical applications** (like acne detection):
- **Prefer higher recall** (better to show a false positive than miss acne)
- Okay with slightly lower precision

### 📈 **Training Curves**

Open `runs/detect/acne_yolov10/results.png` to see:

**1. Loss Curves**:
- Should decrease over time
- Validation loss should track training loss
- If validation loss increases while training decreases → overfitting

**2. mAP Curves**:
- Should increase over time
- Best model is saved at peak mAP

**3. Precision/Recall Curves**:
- Shows model performance across confidence thresholds

### 🎨 **Visual Results**

**Confusion Matrix** (`confusion_matrix.png`):
- Shows what classes are confused
- Diagonal = correct predictions
- Off-diagonal = errors

**Example**:
```
             Predicted
           C   P   Pu  N
Actual C  89   5   3   1   (Comedone)
       P   4  76   8   2   (Papule)
       Pu  2   6  82   1   (Pustule)
       N   1   3   2  85   (Nodule)
```

**Prediction Examples** (`val_batch0_pred.jpg`):
- Shows actual predictions on validation images
- Green box = correct detection
- Red box = incorrect or missed

### ⚕️ **Severity Assessment**

The inference script automatically assesses severity:

```
Severity: MODERATE
Score: 42
Breakdown:
  - Comedones: 12
  - Papules: 8
  - Pustules: 5
  - Nodules: 0
```

**Severity levels**:
- **Clear**: 0 lesions
- **Mild**: 1-20 lesions
- **Moderate**: 21-50 lesions
- **Severe**: >50 lesions OR any nodules

---

## 9. Troubleshooting {#troubleshooting}

### ❌ **Common Errors & Solutions**

#### **Error**: "CUDA out of memory"

**Cause**: Batch size too large for GPU

**Solutions**:
```bash
# Reduce batch size
python scripts/train_yolov10.py --batch 8  # or even 4

# Or reduce image size
python scripts/train_yolov10.py --imgsz 416

# Or train on CPU (slower)
python scripts/train_yolov10.py --device cpu
```

#### **Error**: "YOLOv10 not found" or "cannot import YOLOv10"

**Cause**: YOLOv10 not installed properly

**Solution**:
```bash
# Reinstall YOLOv10
pip uninstall yolov10
pip install git+https://github.com/THU-MIG/yolov10.git

# Verify
python -c "from ultralytics import YOLOv10; print('OK')"
```

#### **Error**: "Dataset not found" or "Path does not exist"

**Cause**: Dataset not prepared or config file incorrect

**Solution**:
```bash
# Check if dataset exists
ls data/yolo_dataset/images/train/

# If empty, prepare dataset
python scripts/prepare_yolo_dataset.py

# Check config file
cat config/yolov10_acne.yaml
```

#### **Error**: Training loss is NaN

**Cause**: Learning rate too high or corrupted data

**Solutions**:
```bash
# Reduce learning rate (edit train script or use smaller model)
python scripts/train_yolov10.py --model yolov10n.pt

# Check for corrupted images
python -c "
import cv2
from pathlib import Path
for img_path in Path('data/yolo_dataset/images/train/').glob('*.jpg'):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'Corrupted: {img_path}')
"
```

#### **Problem**: Training is very slow

**Causes & Solutions**:

1. **Training on CPU**:
   ```bash
   # Use Google Colab for free GPU
   # Or check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Too many workers**:
   ```bash
   # Reduce workers
   python scripts/train_yolov10.py --workers 4
   ```

3. **Not caching images**:
   ```bash
   # Cache images in RAM (needs 8GB+ RAM)
   python scripts/train_yolov10.py --cache
   ```

#### **Problem**: Low accuracy after training

**Possible causes**:

1. **Not enough training data**:
   - Need at least 500+ images
   - Collect more data or use data augmentation

2. **Poor quality labels**:
   - Review and fix label annotations
   - Ensure boxes tightly fit acne lesions

3. **Imbalanced classes**:
   - Check class distribution
   - Collect more examples of rare classes

4. **Not trained long enough**:
   - Increase epochs: `--epochs 200`
   - Check if loss is still decreasing

5. **Model too small**:
   - Try larger model: `--model yolov10m.pt`

### 🔍 **Debugging Tips**

**Check dataset**:
```bash
# Count images
ls data/yolo_dataset/images/train/ | wc -l

# Count labels
ls data/yolo_dataset/labels/train/ | wc -l

# Should be equal!
```

**Validate labels**:
```bash
# Check label format
head data/yolo_dataset/labels/train/img1.txt

# Should show: class_id x_center y_center width height
# All values should be between 0 and 1
```

**Test model on single image**:
```bash
# Quick sanity check
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/acne_sample_1.jpg \
  --show
```

---

## 10. Next Steps {#next-steps}

### 🎓 **Beginner → Intermediate**

1. **Experiment with different models**:
   ```bash
   # Try YOLOv10s (larger, more accurate)
   python scripts/train_yolov10.py --model yolov10s.pt

   # Try YOLOv10m (even better)
   python scripts/train_yolov10.py --model yolov10m.pt
   ```

2. **Tune hyperparameters**:
   - Adjust learning rate, batch size, augmentation
   - See `config/yolov10_acne.yaml` for all options

3. **Collect more data**:
   - More diverse images = better model
   - Aim for 1000+ images

4. **Improve labeling**:
   - Review incorrect predictions
   - Fix mislabeled images
   - Add hard examples

### 🚀 **Intermediate → Advanced**

1. **Model optimization**:
   - Export to ONNX for faster inference
   - Quantization for mobile deployment
   - TensorRT for maximum GPU speed

2. **Build an app**:
   - Web app with FastAPI
   - Mobile app with TensorFlow Lite
   - Desktop app with PyQt

3. **Ensemble models**:
   - Combine YOLOv10 with other models
   - Use voting or averaging for better accuracy

4. **Active learning**:
   - Use model to find hard examples
   - Label and retrain iteratively

### 📚 **Learning Resources**

**YOLOv10**:
- [Official Paper](https://arxiv.org/abs/2405.14458)
- [GitHub Repository](https://github.com/THU-MIG/yolov10)
- [Ultralytics Docs](https://docs.ultralytics.com/)

**Object Detection**:
- [PyImageSearch YOLO tutorials](https://pyimagesearch.com/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

**Medical AI**:
- [Stanford ML for Healthcare](https://www.youtube.com/playlist?list=PLoROMvodv4rOh7_zNhJR5FWj8d3lQxEUk)

### 🎯 **Project Ideas**

1. **Severity Grading System**:
   - Add more detailed severity assessment
   - Implement IGA (Investigator's Global Assessment) scale

2. **Skincare Recommendation Engine**:
   - Recommend products based on acne type
   - Integrate with skincare database

3. **Progress Tracking**:
   - Track acne improvement over time
   - Generate before/after comparisons

4. **Multi-condition Detection**:
   - Detect other skin conditions (rosacea, eczema)
   - Multi-class classification

5. **Real-time Analysis**:
   - Live webcam acne detection
   - Mobile app integration

---

## 🎉 **Congratulations!**

You've learned how to:
- ✅ Set up a YOLOv10 development environment
- ✅ Prepare acne detection datasets
- ✅ Train custom YOLOv10 models
- ✅ Run inference and interpret results
- ✅ Troubleshoot common issues

### 🤝 **Get Help**

**Questions?**
- Open an issue on GitHub
- Check the [FAQ](../README.md#faq)
- Review the [troubleshooting section](#troubleshooting)

**Found a bug?**
- Report it on GitHub Issues
- Include error message and steps to reproduce

**Want to contribute?**
- Fork the repository
- Submit pull requests
- Share your improvements!

---

## 📄 **Quick Reference Card**

### Installation
```bash
git clone <repo>
cd LesionRec
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install git+https://github.com/THU-MIG/yolov10.git
```

### Prepare Dataset
```bash
python scripts/prepare_yolo_dataset.py
```

### Train Model
```bash
# Basic
python scripts/train_yolov10.py

# Custom
python scripts/train_yolov10.py --model yolov10m.pt --epochs 150 --batch 32
```

### Test Model
```bash
# Single image
python scripts/yolov10_inference.py --model best.pt --source image.jpg

# Directory
python scripts/yolov10_inference.py --model best.pt --source images/

# Save JSON
python scripts/yolov10_inference.py --model best.pt --source image.jpg --save-json results.json
```

### Resume Training
```bash
python scripts/train_yolov10.py --resume runs/detect/acne_yolov10/weights/last.pt
```

---
