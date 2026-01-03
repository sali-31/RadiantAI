# YOLOv8 Acne Detection Guide

## 🎯 Overview

This guide shows you how to use YOLOv8 for acne detection and dataset labeling. YOLOv8 is now your primary detection model, replacing the Roboflow ensemble approach.

**Why YOLOv8?**
- ✅ **Free & Open Source** - No API costs
- ✅ **Fast** - Real-time detection
- ✅ **Customizable** - Full control over training
- ✅ **State-of-the-Art** - Latest YOLO architecture
- ✅ **Easy to Use** - Simple Python API

---

## 📁 Project Files Created

| File | Purpose |
|------|---------|
| [config/yolo_acne.yaml](../config/yolo_acne.yaml) | Dataset configuration (classes, paths, hyperparameters) |
| [scripts/prepare_yolo_dataset.py](../scripts/prepare_yolo_dataset.py) | Prepare dataset for YOLO training |
| [scripts/train_yolo.py](../scripts/train_yolo.py) | Train YOLOv8 model |
| [scripts/yolo_inference.py](../scripts/yolo_inference.py) | Run inference and generate labels |

---

## 🚀 Quick Start (3 Steps)

### **Step 1: Prepare Your Dataset**

```bash
# Option A: Use pre-trained YOLO to generate initial labels (recommended)
python scripts/prepare_yolo_dataset.py \
  --source data/raw/acne \
  --method pretrained \
  --model-size n

# Option B: Create empty labels for manual annotation
python scripts/prepare_yolo_dataset.py \
  --source data/raw/acne \
  --method empty
```

**What this does:**
- Creates `data/yolo_dataset/` folder with YOLO structure
- Splits data into train (70%), val (15%), test (15%)
- Generates initial labels using pre-trained YOLOv8 (if `--method pretrained`)

**Output:**
```
data/yolo_dataset/
├── images/
│   ├── train/  (1,883 images)
│   ├── val/    (404 images)
│   └── test/   (403 images)
└── labels/
    ├── train/  (1,883 .txt files)
    ├── val/    (404 .txt files)
    └── test/   (403 .txt files)
```

### **Step 2: Train YOLOv8**

```bash
# Basic training (nano model, 100 epochs)
python scripts/train_yolo.py

# Advanced training (small model, 200 epochs, larger batch)
python scripts/train_yolo.py \
  --model yolov8s.pt \
  --epochs 200 \
  --batch 32 \
  --patience 30
```

**Training time estimates:**
- **YOLOv8n** (nano): ~30-45 min on GPU, ~3-4 hours on CPU
- **YOLOv8s** (small): ~1-1.5 hours on GPU, ~8-10 hours on CPU
- **YOLOv8m** (medium): ~2-3 hours on GPU, ~20+ hours on CPU

**What this does:**
- Downloads pre-trained weights
- Fine-tunes on your acne dataset
- Saves checkpoints to `runs/detect/acne_detector/`
- Creates best.pt and last.pt model files

### **Step 3: Generate Labels for Your Dataset**

```bash
# Generate labels in YOLO format + CSV like skin_features.csv
python scripts/yolo_inference.py \
  --model runs/detect/acne_detector/weights/best.pt \
  --source data/raw/acne \
  --save-labels \
  --save-csv data/labels/acne_yolo_labels.csv \
  --save-vis
```

**Output:**
- `runs/inference/labels/` - YOLO format labels (.txt files)
- `data/labels/acne_yolo_labels.csv` - CSV with features (like skin_features.csv)
- `runs/inference/visualizations/` - Images with bounding boxes drawn

---

## 📊 Understanding YOLO Label Format

### **YOLO Format Explained**

Each line in a `.txt` label file represents one detection:

```
<class_id> <x_center> <y_center> <width> <height>
```

**All coordinates are normalized to [0, 1]**

**Example:** `0 0.503 0.421 0.135 0.087`
- **0** = class_id (comedone)
- **0.503** = x_center (50.3% from left)
- **0.421** = y_center (42.1% from top)
- **0.135** = width (13.5% of image width)
- **0.087** = height (8.7% of image height)

### **Class IDs**

From [config/yolo_acne.yaml](../config/yolo_acne.yaml):

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | comedone | Blackheads, whiteheads |
| 1 | papule | Red inflamed bumps |
| 2 | pustule | Pus-filled lesions |
| 3 | nodule | Large deep lesions |

---

## 🎓 Detailed Workflow

### **Workflow 1: Training from Scratch**

If you don't have labeled data:

```bash
# 1. Generate initial labels with pre-trained YOLO
python scripts/prepare_yolo_dataset.py \
  --source data/raw/acne \
  --method pretrained

# 2. (Optional) Manually refine labels
#    Use tools like LabelImg, Roboflow, or CVAT

# 3. Train on your dataset
python scripts/train_yolo.py --epochs 100

# 4. Generate final labels with trained model
python scripts/yolo_inference.py \
  --model runs/detect/acne_detector/weights/best.pt \
  --source data/raw/acne \
  --save-labels \
  --save-csv data/labels/acne_labels.csv
```

### **Workflow 2: If You Have Existing Labels**

If you already have annotations in another format (JSON, XML, etc.):

```bash
# 1. Convert your labels to YOLO format
#    (You'll need to write a custom conversion script)

# 2. Manually organize into YOLO structure
#    data/yolo_dataset/images/{train,val,test}
#    data/yolo_dataset/labels/{train,val,test}

# 3. Train
python scripts/train_yolo.py
```

### **Workflow 3: Using Roboflow to Bootstrap**

Combine Roboflow + YOLOv8:

```bash
# 1. Use Roboflow ensemble to label a small sample (100-200 images)
python scripts/generate_labels.py \
  --api-key YOUR_ROBOFLOW_KEY \
  --input data/raw/acne \
  --output data/labels/roboflow_initial.csv

# 2. Convert Roboflow results to YOLO format
#    (Write custom script to parse bounding boxes from CSV)

# 3. Train YOLOv8 on Roboflow-labeled data
python scripts/train_yolo.py --epochs 50

# 4. Use trained YOLO to label remaining images
python scripts/yolo_inference.py \
  --model runs/detect/acne_detector/weights/best.pt \
  --source data/raw/acne \
  --save-labels
```

---

## ⚙️ Model Selection Guide

Choose based on your needs:

| Model | Params | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| **yolov8n** | 3.2M | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Fast iterations, testing |
| **yolov8s** | 11.2M | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended for production** |
| **yolov8m** | 25.9M | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy needed |
| **yolov8l** | 43.7M | ⚡⚡ | ⭐⭐⭐⭐⭐ | Research, best performance |
| **yolov8x** | 68.2M | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |

**Recommendation:** Start with **yolov8n** for fast experimentation, then train **yolov8s** for production.

---

## 📈 Monitoring Training

### **TensorBoard**

View training progress in real-time:

```bash
tensorboard --logdir runs/detect/acne_detector
```

Then open: http://localhost:6006

**Metrics to watch:**
- **mAP50** - Mean Average Precision at IoU 0.5 (target: >0.70)
- **mAP50-95** - mAP across IoU thresholds (target: >0.50)
- **Precision** - How many detections are correct (target: >0.85)
- **Recall** - How many acne lesions are found (target: >0.80)
- **Loss** - Should decrease over time

### **Weights & Biases Integration**

For advanced tracking:

```python
# In scripts/train_yolo.py, add:
import wandb
wandb.init(project="lesionrec-acne", name="yolov8s-run1")

# YOLO will automatically log to W&B
```

---

## 🔍 Evaluating Your Model

### **Validation Metrics**

After training:

```bash
# Evaluate on test set
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/acne_detector/weights/best.pt'); metrics = model.val(data='config/yolo_acne.yaml', split='test'); print(metrics.box.map50)"
```

### **Visual Inspection**

```bash
# Generate predictions with visualizations
python scripts/yolo_inference.py \
  --model runs/detect/acne_detector/weights/best.pt \
  --source data/yolo_dataset/images/test \
  --save-vis
```

Then manually review `runs/inference/visualizations/`

### **Error Analysis**

Common issues:

| Problem | Cause | Solution |
|---------|-------|----------|
| Low recall | Model misses acne | Lower confidence threshold, train longer |
| Low precision | Too many false positives | Increase confidence threshold, add hard negatives |
| Model overfits | High train accuracy, low val accuracy | More data augmentation, reduce model size |
| Training stalls | Learning rate too high/low | Adjust `lr0` in config |

---

## 💡 Tips & Best Practices

### **Data Augmentation**

Already configured in [config/yolo_acne.yaml](../config/yolo_acne.yaml):
- ✅ Horizontal flip (important for facial symmetry)
- ✅ Rotation (±15°)
- ✅ Color jitter (HSV augmentation for skin tones)
- ✅ Mosaic augmentation (combines 4 images)

### **Class Imbalance**

If some acne types are rare:

```python
# In training script, add class weights
model.train(
    data='config/yolo_acne.yaml',
    epochs=100,
    cls=1.0,  # Increase classification loss weight
    # ... other params
)
```

### **Transfer Learning**

Start from a domain-specific checkpoint:

```bash
# If someone shares a skin-disease-trained model
python scripts/train_yolo.py \
  --model path/to/skin_disease.pt \
  --epochs 50
```

### **Confidence Thresholds**

For inference:

| Use Case | Confidence Threshold |
|----------|---------------------|
| High precision (few false positives) | 0.5 - 0.7 |
| Balanced | 0.25 - 0.4 |
| High recall (catch all acne) | 0.1 - 0.2 |

---

## 🔄 Comparing YOLOv8 vs Roboflow

| Feature | YOLOv8 | Roboflow Ensemble |
|---------|--------|-------------------|
| **Cost** | Free | Free tier + paid |
| **Speed** | Very fast (local GPU) | Medium (API latency) |
| **Customization** | Full control | Limited |
| **Training data** | Your own | Pre-trained |
| **Deployment** | Local or cloud | Cloud only |
| **Best for** | Production, high volume | Quick prototyping |

**Hybrid Approach:**
1. Use Roboflow to label initial dataset (100-500 images)
2. Train YOLOv8 on Roboflow-labeled data
3. Use YOLOv8 for production inference

---

## 📦 Exporting Models

### **ONNX (For Production)**

```python
from ultralytics import YOLO

model = YOLO('runs/detect/acne_detector/weights/best.pt')
model.export(format='onnx')  # Creates best.onnx
```

### **TensorFlow Lite (For Mobile)**

```python
model.export(format='tflite')  # Creates best-int8.tflite
```

### **CoreML (For iOS)**

```python
model.export(format='coreml')  # Creates best.mlpackage
```

---

## 🐛 Troubleshooting

### **Out of Memory Error**

```bash
# Reduce batch size
python scripts/train_yolo.py --batch 8

# Or use smaller model
python scripts/train_yolo.py --model yolov8n.pt --batch 16
```

### **Slow Training on CPU**

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, training will be slow (use Google Colab or smaller dataset)
```

### **Poor Performance**

1. **Check data quality**
   - Are labels correct?
   - Is there enough data? (min 500-1000 images)
   - Is data diverse? (different skin tones, lighting)

2. **Adjust hyperparameters**
   - Increase epochs: `--epochs 200`
   - Increase patience: `--patience 30`
   - Try different model sizes

3. **Review training curves**
   - Use TensorBoard to diagnose

---

## 📚 Additional Resources

### **Official Docs**
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Training Custom Datasets](https://docs.ultralytics.com/modes/train/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)

### **Tutorials**
- [YOLOv8 Custom Training Tutorial](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)
- [Medical Image Object Detection](https://pyimagesearch.com/2020/06/29/opencv-object-detection-in-medical-images/)

### **Annotation Tools**
- [LabelImg](https://github.com/heartexlabs/labelImg) - Desktop tool
- [Roboflow](https://roboflow.com/) - Web-based
- [CVAT](https://cvat.org/) - Advanced annotation platform

---

## 🎯 Next Steps

1. **Prepare dataset:** `python scripts/prepare_yolo_dataset.py --source data/raw/acne`
2. **Train model:** `python scripts/train_yolo.py`
3. **Generate labels:** `python scripts/yolo_inference.py --model runs/detect/acne_detector/weights/best.pt --source data/raw/acne --save-csv data/labels/acne_yolo.csv`
4. **Integrate into app:** Use trained model in your FastAPI backend

---

## ✅ Summary

**You now have a complete YOLOv8 pipeline for acne detection!**

- ✅ Dataset preparation script
- ✅ Training script
- ✅ Inference/labeling script
- ✅ Configuration files
- ✅ Complete documentation

**Total time to first model: ~1-2 hours** (including training)

Good luck with your acne detection project! 🚀
