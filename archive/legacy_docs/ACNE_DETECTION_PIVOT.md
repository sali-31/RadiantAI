# Acne Detection Pivot - Updated Strategy

## Executive Summary

**Pivot Decision**: Focus exclusively on acne detection instead of general skin lesion detection.

**Current Approach**: Using 3 pre-trained Roboflow models in an ensemble:
1. `acnedet-v1` - Best for crisp, head-focused images
2. `skin_disease_ak` - Good classifier for "Acne and Rosacea Photos"
3. `skn-1` - Backup detector when acnedet-v1 fails

## Updated Dataset Strategy

### Primary Datasets (KEEP)

| Dataset | Images | Use Case | Priority |
|---------|--------|----------|----------|
| **Acne Dataset** | ~1,800 | Core training/validation | 🔴 HIGH |
| **Acne-Wrinkles-Spots** | ~500+ acne | Supplementary training | 🟡 MEDIUM |
| **Skin Disease Dataset** | ~500 (filtered) | Additional acne cases | 🟡 MEDIUM |
| **FitzPatrick17k** | ~100 (filtered) | Bias/diversity testing | 🟢 LOW |

### Datasets to Remove

❌ **HAM10000** - Melanoma/cancer focused, not acne
❌ **ISIC Archive** - Skin cancer focused, not acne

### Data Preparation Plan

```bash
# 1. Download acne-specific datasets
python scripts/download_datasets.py --dataset acne_primary
python scripts/download_datasets.py --dataset acne_spots
python scripts/download_datasets.py --dataset fitzpatrick

# 2. Filter broad datasets for acne only
python scripts/filter_datasets.py \
  --input data/raw/skin_disease \
  --output data/processed/skin_disease_acne_only \
  --labels "acne,rosacea,comedone,papule,pustule"

# 3. Create unified dataset
python scripts/create_unified_dataset.py \
  --output data/processed/acne_unified \
  --split 0.7/0.15/0.15
```

## Ensemble Model Strategy

### Current Model Performance Analysis

Based on your observations:

| Model | Strengths | Weaknesses | Use When |
|-------|-----------|------------|----------|
| **acnedet-v1** | High precision on crisp head shots | Fails on unclear/distant images | Primary detector |
| **skin_disease_ak** | Good classifier for acne/rosacea | Over-detects other conditions | Confidence scorer |
| **skn-1** | Handles varied image quality | Lower precision | Fallback detector |

### Recommended Ensemble Logic

```python
def ensemble_acne_detection(image):
    """
    Smart ensemble combining 3 Roboflow models
    """
    # Step 1: Try primary detector
    result_acnedet = acnedet_v1.predict(image)

    # Step 2: Get classification confidence
    result_classifier = skin_disease_ak.predict(image)

    # Step 3: Use fallback if needed
    result_fallback = skn_1.predict(image)

    # Decision logic
    if result_acnedet.detections > 0:
        # acnedet-v1 found something - use it
        if result_classifier.label == "Acne and Rosacea Photos":
            # High confidence - use acnedet results
            return {
                'detections': result_acnedet.detections,
                'count': len(result_acnedet.detections),
                'confidence': 'high',
                'source': 'acnedet-v1'
            }
        else:
            # Classifier disagrees - be cautious
            return {
                'detections': result_acnedet.detections,
                'count': len(result_acnedet.detections),
                'confidence': 'medium',
                'source': 'acnedet-v1',
                'warning': f'Classifier detected: {result_classifier.label}'
            }

    elif result_classifier.label == "Acne and Rosacea Photos" and result_classifier.confidence > 0.85:
        # No detections but high classification confidence - use fallback
        if result_fallback.detections > 0:
            return {
                'detections': result_fallback.detections,
                'count': len(result_fallback.detections),
                'confidence': 'medium',
                'source': 'skn-1 (fallback)'
            }
        else:
            return {
                'detections': [],
                'count': 0,
                'confidence': 'low',
                'source': 'classifier-only',
                'note': 'Classified as acne but no specific lesions detected'
            }

    else:
        # Nothing found
        return {
            'detections': [],
            'count': 0,
            'confidence': 'none',
            'source': 'no-detection'
        }
```

## Addressing Current Issues

### Issue 1: False Positives from skin_disease_ak

**Problem**: "Viral infections" label on image that has acne

**Solutions**:
1. **Whitelist approach**: Only accept these labels:
   ```python
   VALID_ACNE_LABELS = {
       "Acne and Rosacea Photos",
       "Acne Vulgaris",
       "Comedones",
       "Papules",
       "Pustules"
   }

   if result.label not in VALID_ACNE_LABELS:
       # Ignore this result
       pass
   ```

2. **Confidence threshold**: Increase minimum confidence
   ```python
   MIN_CONFIDENCE = 0.85  # Increase from default
   ```

3. **Use as validator only**: Don't use for detection, only for validation
   ```python
   # Use skin_disease_ak to CONFIRM acnedet-v1, not replace it
   ```

### Issue 2: Inconsistent Detection Quality

**Root Cause**: Image quality varies (crisp head shots vs. distant/blurry)

**Solutions**:

1. **Image Quality Preprocessing**:
   ```python
   def preprocess_for_detection(image):
       """Enhance image quality before detection"""
       import cv2

       # 1. Enhance contrast
       lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
       l, a, b = cv2.split(lab)
       clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
       l = clahe.apply(l)
       enhanced = cv2.merge([l, a, b])
       enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

       # 2. Denoise
       denoised = cv2.fastNlMeansDenoisingColored(enhanced)

       # 3. Sharpen
       kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
       sharpened = cv2.filter2D(denoised, -1, kernel)

       return sharpened
   ```

2. **Multi-Resolution Detection**:
   ```python
   def multi_scale_detection(image):
       """Run detection at multiple scales"""
       results = []

       for scale in [0.8, 1.0, 1.2]:
           scaled = cv2.resize(image, None, fx=scale, fy=scale)
           result = acnedet_v1.predict(scaled)
           results.append(result)

       # Merge results with NMS
       return merge_detections(results)
   ```

3. **Image Quality Scorer**:
   ```python
   def assess_image_quality(image):
       """Determine which model to prioritize"""
       import cv2

       # Calculate sharpness (Laplacian variance)
       gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
       sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

       # Calculate brightness
       brightness = np.mean(image)

       if sharpness > 100 and 50 < brightness < 200:
           return "high_quality"  # Use acnedet-v1
       else:
           return "low_quality"   # Use skn-1 or fallback
   ```

## Fine-Tuning Recommendations

### Option 1: Fine-Tune Roboflow Models (Recommended)

Since you're already using Roboflow, leverage their platform:

1. **Upload your acne dataset to Roboflow**:
   - Combine Acne Dataset + filtered datasets
   - ~2,000-3,000 images total

2. **Fine-tune acnedet-v1**:
   ```bash
   # Use Roboflow's fine-tuning API
   # This improves on your specific use case
   ```

3. **Benefits**:
   - Maintains Roboflow infrastructure
   - Easy deployment
   - Version control built-in

### Option 2: Train Custom YOLOv8 Model

For more control, train from scratch:

```python
from ultralytics import YOLO

# 1. Prepare dataset in YOLO format
# data.yaml:
# train: data/processed/acne_unified/train/images
# val: data/processed/acne_unified/val/images
# test: data/processed/acne_unified/test/images
# names: ['comedone', 'papule', 'pustule', 'nodule']

# 2. Train YOLOv8
model = YOLO('yolov8n.pt')  # Start with nano for speed

model.train(
    data='config/acne_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    augment=True,
    device=0,  # GPU
    project='acne_detection',
    name='yolov8n_acne_v1'
)

# 3. Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# 4. Export for deployment
model.export(format='onnx')  # For FastAPI serving
```

### Option 3: Hybrid Approach (BEST)

Combine Roboflow models with custom post-processing:

1. Keep using Roboflow for inference (fast, reliable)
2. Add custom fusion logic (your ensemble code)
3. Fine-tune on your specific acne dataset
4. Add image enhancement preprocessing

## Updated Data Download Script

```python
# scripts/download_acne_datasets.py

def download_acne_datasets():
    """Download acne-specific datasets"""

    datasets = {
        'acne_primary': {
            'url': 'https://www.kaggle.com/datasets/nayanchaure/acne-dataset',
            'size': '~200MB',
            'images': 1800,
            'type': 'kaggle'
        },
        'acne_spots': {
            'url': 'https://www.kaggle.com/datasets/ranvijaybalbir/acne-wrinkles-spots-classification',
            'size': '~150MB',
            'images': 500,
            'type': 'kaggle'
        },
        'skin_disease': {
            'url': 'https://www.kaggle.com/datasets/pacificrm/skindiseasedataset',
            'size': '~500MB',
            'images': 5000,
            'type': 'kaggle',
            'filter': 'acne,rosacea'  # Only extract these
        },
        'fitzpatrick': {
            'url': 'https://github.com/mattgroh/fitzpatrick17k',
            'size': '~2GB',
            'images': 100,
            'type': 'github',
            'filter': 'acne'
        }
    }

    # Implementation...
```

## Validation Strategy

### Create Comprehensive Test Set

```python
# Mix of:
# - Different skin tones (FitzPatrick17k)
# - Different image qualities (crisp vs. blurry)
# - Different acne severities (mild, moderate, severe)
# - Different angles (frontal, side, close-up)

test_set_composition = {
    'crisp_headshots': 100,      # Where acnedet-v1 excels
    'low_quality': 50,           # Where skn-1 is needed
    'diverse_skin_tones': 75,    # From FitzPatrick17k
    'severe_acne': 25,           # Edge cases
    'mild_acne': 50,             # Challenging detections
}
```

### Metrics to Track

```python
metrics = {
    # Detection metrics
    'precision': 'TP / (TP + FP)',
    'recall': 'TP / (TP + FN)',
    'f1_score': '2 * (precision * recall) / (precision + recall)',
    'mAP50': 'Mean average precision at IoU 0.5',

    # Ensemble metrics
    'ensemble_agreement_rate': 'How often models agree',
    'fallback_usage_rate': 'How often skn-1 is used',
    'false_positive_rate': 'Non-acne detected as acne',

    # Fairness metrics
    'accuracy_by_skin_tone': 'Performance across Fitzpatrick scale',
    'detection_bias': 'Variance across demographics'
}
```

### 1. **Model Fusion Best Practices**

```python
# Don't just concatenate results - use weighted voting
def weighted_ensemble(results, weights):
    """
    results: List of model outputs
    weights: Confidence weights for each model
    """
    from collections import defaultdict

    # Group overlapping detections
    grouped = group_overlapping_boxes(results, iou_threshold=0.5)

    # Vote with weights
    final_detections = []
    for group in grouped:
        weighted_score = sum(det.confidence * weights[det.model]
                            for det in group)
        if weighted_score > threshold:
            final_detections.append(merge_boxes(group))

    return final_detections
```

### 2. **Handle Edge Cases**

```python
edge_cases = {
    'no_face_detected': 'Use full-image analysis',
    'multiple_faces': 'Analyze each face separately',
    'partial_face': 'Adjust ROI for acne-prone areas',
    'low_light': 'Apply brightness enhancement',
    'high_makeup': 'Warn user or use texture analysis'
}
```

### 3. **Log Everything for Analysis**

```python
# Save predictions for later analysis
prediction_log = {
    'image_id': '001',
    'timestamp': '2025-01-02T10:30:00',
    'models_used': ['acnedet-v1', 'skn-1'],
    'primary_result': result_acnedet,
    'fallback_result': result_skn1,
    'final_decision': ensemble_result,
    'image_quality_score': 85.3,
    'preprocessing_applied': ['clahe', 'denoise'],
    'confidence': 0.92
}
```

### 4. **Build Confidence Calibration**

```python
# Calibrate ensemble confidence based on agreement
def calibrate_confidence(results):
    model_agreement = calculate_agreement(results)

    if model_agreement > 0.9:
        return 'high'
    elif model_agreement > 0.7:
        return 'medium'
    else:
        return 'low'
```

### 5. **Create Model Performance Dashboard**

Track which model is used when:
```python
usage_stats = {
    'acnedet-v1_success': 450,  # 45% of time
    'skn-1_fallback': 350,      # 35% of time
    'no_detection': 200,        # 20% of time
    'avg_confidence': 0.83,
    'false_positive_rate': 0.12
}
```

## Additional Resources

### Acne Detection Research Papers
- "Deep Learning for Acne Vulgaris Detection" (2023)
- "Multi-class Skin Lesion Classification using CNNs"

### Data Augmentation for Acne
```python
import albumentations as A

acne_augmentation = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.RandomGamma(p=0.3),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    A.CLAHE(p=0.3),  # Enhance local contrast
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.5
    )
])
```

## Questions to Consider

1. **Acne Types**: Are you detecting all acne types (comedones, papules, pustules, nodules) or just presence/absence?

2. **Severity Grading**: Do you need to classify severity (mild, moderate, severe)?

3. **Region Detection**: Should you detect T-zone, cheeks, jawline separately?

4. **Temporal Analysis**: Will users upload multiple images over time to track progress?

5. **Product Recommendations**: How will detection output map to product recommendations?
   - Mild acne → gentle salicylic acid cleanser
   - Moderate acne → benzoyl peroxide + moisturizer
   - Severe → recommend dermatologist visit

## Success Criteria

Define clear metrics for your pivot:

```yaml
success_metrics:
  detection:
    - precision: > 0.85
    - recall: > 0.80
    - f1_score: > 0.82
    - inference_time: < 500ms

  fairness:
    - accuracy_variance_across_skin_tones: < 10%
    - no_significant_bias: true

  user_experience:
    - detection_confidence_shown: true
    - false_positive_rate: < 15%
    - user_satisfaction: > 4.0/5
```

---
