"""
Ensemble Acne Detection using Multiple Roboflow Models

This module implements the smart ensemble logic combining:
1. acnedet-v1 - Primary detector (best for crisp head shots)
2. skin_disease_ak - Classifier validator
3. skn-1 - Fallback detector

Usage:
    from ensemble_detector import AcneEnsembleDetector

    detector = AcneEnsembleDetector(
        api_key="YOUR_ROBOFLOW_API_KEY"
    )

    result = detector.detect(image_path="acne.jpg")
    print(f"Acne count: {result['count']}")
    print(f"Confidence: {result['confidence']}")
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single acne detection"""
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    class_name: str
    model_source: str


@dataclass
class EnsembleResult:
    """Result from ensemble detection"""
    detections: List[Detection]
    count: int
    confidence_level: str  # 'high', 'medium', 'low', 'none'
    primary_model: str
    classification_label: Optional[str] = None
    classification_confidence: Optional[float] = None
    image_quality: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ImageQualityAssessor:
    """Assess image quality to determine optimal detection strategy"""

    @staticmethod
    def assess(image: np.ndarray) -> Dict[str, Union[float, str]]:
        """
        Assess image quality metrics

        Returns:
            dict with sharpness, brightness, and quality level
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # Calculate brightness
        brightness = np.mean(image)

        # Calculate contrast (standard deviation)
        contrast = np.std(gray)

        # Determine quality level
        if sharpness > 100 and 50 < brightness < 200 and contrast > 30:
            quality = "high"
        elif sharpness > 50 and 30 < brightness < 220 and contrast > 20:
            quality = "medium"
        else:
            quality = "low"

        return {
            'sharpness': float(sharpness),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'quality': quality
        }


class ImagePreprocessor:
    """Preprocess images to improve detection quality"""

    @staticmethod
    def enhance(image: np.ndarray, quality: str = "medium") -> np.ndarray:
        """
        Enhance image based on quality assessment

        Args:
            image: Input image
            quality: Quality level ('high', 'medium', 'low')

        Returns:
            Enhanced image
        """
        if quality == "high":
            # Minimal processing for high-quality images
            return image

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        if quality == "low":
            # Additional processing for low-quality images
            # Denoise
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

            # Sharpen
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced


class AcneEnsembleDetector:
    """
    Ensemble detector combining multiple Roboflow models
    """

    # Valid acne-related labels from skin_disease_ak model
    VALID_ACNE_LABELS = {
        "Acne and Rosacea Photos",
        "Acne Vulgaris",
        "Comedones",
        "Papules",
        "Pustules",
        "Nodules"
    }

    # Acne severity mapping
    ACNE_TYPE_SEVERITY = {
        "comedone": "mild",
        "whitehead": "mild",
        "blackhead": "mild",
        "papule": "moderate",
        "pustule": "moderate",
        "nodule": "severe",
        "cyst": "severe"
    }

    def __init__(
        self,
        api_key: str,
        acnedet_model: str = "acnedet/acnedet-v1/2",
        skin_disease_model: str = "kelixo/skin_disease_ak/1",
        skn_model: str = "skn-f1vaw/skn-1/2",
        confidence_threshold: float = 0.4,
        classification_threshold: float = 0.85
    ):
        """
        Initialize ensemble detector

        Args:
            api_key: Roboflow API key
            acnedet_model: Model identifier for acnedet-v1
            skin_disease_model: Model identifier for skin_disease_ak
            skn_model: Model identifier for skn-1
            confidence_threshold: Minimum confidence for detections
            classification_threshold: Minimum confidence for classification
        """
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.classification_threshold = classification_threshold

        # Initialize models (lazy loading)
        self._acnedet = None
        self._skin_disease = None
        self._skn = None

        self.acnedet_model = acnedet_model
        self.skin_disease_model = skin_disease_model
        self.skn_model = skn_model

        self.quality_assessor = ImageQualityAssessor()
        self.preprocessor = ImagePreprocessor()

        logger.info("Initialized AcneEnsembleDetector")

    def _load_models(self):
        """Lazy load Roboflow models"""
        if self._acnedet is None:
            from roboflow import Roboflow

            rf = Roboflow(api_key=self.api_key)

            # Load models
            logger.info("Loading Roboflow models...")

            try:
                project1 = rf.workspace().project(self.acnedet_model.split('/')[0])
                self._acnedet = project1.version(
                    self.acnedet_model.split('/')[-1]
                ).model

                project2 = rf.workspace().project(self.skin_disease_model.split('/')[0])
                self._skin_disease = project2.version(
                    self.skin_disease_model.split('/')[-1]
                ).model

                project3 = rf.workspace().project(self.skn_model.split('/')[0])
                self._skn = project3.version(
                    self.skn_model.split('/')[-1]
                ).model

                logger.info("✓ All models loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        preprocess: bool = True,
        return_visualization: bool = False
    ) -> Union[EnsembleResult, Tuple[EnsembleResult, np.ndarray]]:
        """
        Detect acne using ensemble approach

        Args:
            image: Path to image or numpy array
            preprocess: Whether to preprocess image
            return_visualization: Whether to return annotated image

        Returns:
            EnsembleResult or (EnsembleResult, annotated_image)
        """
        # Load models if not already loaded
        if self._acnedet is None:
            self._load_models()

        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        # Assess image quality
        quality_info = self.quality_assessor.assess(img)
        logger.info(f"Image quality: {quality_info['quality']} "
                   f"(sharpness: {quality_info['sharpness']:.1f}, "
                   f"brightness: {quality_info['brightness']:.1f})")

        # Preprocess if needed
        if preprocess and quality_info['quality'] != "high":
            img_processed = self.preprocessor.enhance(img, quality_info['quality'])
        else:
            img_processed = img

        # Step 1: Run primary detector (acnedet-v1)
        logger.info("Running acnedet-v1 (primary detector)...")
        result_acnedet = self._acnedet.predict(img_processed, confidence=self.confidence_threshold)

        # Step 2: Run classifier (skin_disease_ak)
        logger.info("Running skin_disease_ak (classifier)...")
        result_classifier = self._skin_disease.predict(img_processed, confidence=self.classification_threshold)

        # Step 3: Run fallback detector (skn-1)
        logger.info("Running skn-1 (fallback detector)...")
        result_skn = self._skn.predict(img_processed, confidence=self.confidence_threshold)

        # Apply ensemble logic
        ensemble_result = self._apply_ensemble_logic(
            result_acnedet,
            result_classifier,
            result_skn,
            quality_info
        )

        if return_visualization:
            annotated = self._visualize_results(img, ensemble_result)
            return ensemble_result, annotated

        return ensemble_result

    def _apply_ensemble_logic(
        self,
        result_acnedet,
        result_classifier,
        result_skn,
        quality_info: Dict
    ) -> EnsembleResult:
        """
        Apply smart ensemble logic to combine model outputs

        Decision tree:
        1. If acnedet-v1 detects acne:
           - If classifier agrees → HIGH confidence
           - If classifier disagrees → MEDIUM confidence (with warning)
        2. If acnedet-v1 finds nothing but classifier is confident:
           - Use skn-1 as fallback
        3. If all fail → NO DETECTION
        """
        detections = []
        warnings = []

        # Parse classifier results
        classifier_label = None
        classifier_confidence = 0.0

        if hasattr(result_classifier, 'predictions') and result_classifier.predictions:
            pred = result_classifier.predictions[0]
            classifier_label = pred.get('class', pred.get('predicted_class'))
            classifier_confidence = pred.get('confidence', 0.0)

        logger.info(f"Classifier: {classifier_label} ({classifier_confidence:.2f})")

        # Parse acnedet-v1 results
        acnedet_detections = []
        if hasattr(result_acnedet, 'predictions') and result_acnedet.predictions:
            for pred in result_acnedet.predictions:
                det = Detection(
                    bbox=(pred['x'], pred['y'], pred['width'], pred['height']),
                    confidence=pred['confidence'],
                    class_name=pred['class'],
                    model_source='acnedet-v1'
                )
                acnedet_detections.append(det)

        logger.info(f"acnedet-v1 detections: {len(acnedet_detections)}")

        # Parse skn-1 results
        skn_detections = []
        if hasattr(result_skn, 'predictions') and result_skn.predictions:
            for pred in result_skn.predictions:
                det = Detection(
                    bbox=(pred['x'], pred['y'], pred['width'], pred['height']),
                    confidence=pred['confidence'],
                    class_name=pred['class'],
                    model_source='skn-1'
                )
                skn_detections.append(det)

        logger.info(f"skn-1 detections: {len(skn_detections)}")

        # DECISION LOGIC

        # Case 1: acnedet-v1 found acne
        if acnedet_detections:
            detections = acnedet_detections

            # Check if classifier agrees
            if classifier_label in self.VALID_ACNE_LABELS:
                confidence_level = "high"
                primary_model = "acnedet-v1"
                logger.info("✓ High confidence: acnedet-v1 + classifier agreement")
            else:
                confidence_level = "medium"
                primary_model = "acnedet-v1"
                warnings.append(
                    f"Classifier detected '{classifier_label}' instead of acne. "
                    f"Results should be interpreted cautiously."
                )
                logger.warning(f"⚠️ Classifier disagreement: {classifier_label}")

        # Case 2: No detections from acnedet-v1, but classifier is confident
        elif classifier_label in self.VALID_ACNE_LABELS and classifier_confidence > self.classification_threshold:
            logger.info("acnedet-v1 found nothing, using fallback (skn-1)")

            if skn_detections:
                detections = skn_detections
                confidence_level = "medium"
                primary_model = "skn-1 (fallback)"
                warnings.append(
                    "Primary detector (acnedet-v1) found no acne. "
                    "Using fallback detector (skn-1)."
                )
            else:
                detections = []
                confidence_level = "low"
                primary_model = "classifier-only"
                warnings.append(
                    "Classified as acne but no specific lesions detected. "
                    "Possible diffuse acne or low image quality."
                )

        # Case 3: Nothing found
        else:
            detections = []
            confidence_level = "none"
            primary_model = "no-detection"
            logger.info("No acne detected by any model")

        return EnsembleResult(
            detections=detections,
            count=len(detections),
            confidence_level=confidence_level,
            primary_model=primary_model,
            classification_label=classifier_label,
            classification_confidence=classifier_confidence,
            image_quality=quality_info['quality'],
            warnings=warnings
        )

    def _visualize_results(
        self,
        image: np.ndarray,
        result: EnsembleResult
    ) -> np.ndarray:
        """
        Visualize detection results on image

        Args:
            image: Original image
            result: Detection results

        Returns:
            Annotated image
        """
        annotated = image.copy()

        # Draw bounding boxes
        for det in result.detections:
            x, y, w, h = det.bbox
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Color based on confidence
            if result.confidence_level == "high":
                color = (0, 255, 0)  # Green
            elif result.confidence_level == "medium":
                color = (255, 165, 0)  # Orange
            else:
                color = (255, 0, 0)  # Red

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add summary text
        summary = f"Count: {result.count} | Confidence: {result.confidence_level}"
        cv2.putText(annotated, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def assess_severity(self, result: EnsembleResult) -> Dict[str, Union[str, int]]:
        """
        Assess acne severity based on detections

        Returns:
            dict with severity level and type breakdown
        """
        if result.count == 0:
            return {
                'severity': 'none',
                'total_count': 0,
                'type_breakdown': {}
            }

        # Count by type
        type_counts = {}
        for det in result.detections:
            class_name = det.class_name.lower()
            type_counts[class_name] = type_counts.get(class_name, 0) + 1

        # Determine severity
        total = result.count
        has_severe = any(t in type_counts for t in ['nodule', 'cyst'])

        if total > 50 or has_severe:
            severity = 'severe'
        elif total > 20:
            severity = 'moderate'
        else:
            severity = 'mild'

        return {
            'severity': severity,
            'total_count': total,
            'type_breakdown': type_counts
        }


# Example usage
if __name__ == "__main__":
    # Example usage (requires Roboflow API key)
    import sys

    if len(sys.argv) < 3:
        print("Usage: python ensemble_detector.py <api_key> <image_path>")
        sys.exit(1)

    api_key = sys.argv[1]
    image_path = sys.argv[2]

    # Initialize detector
    detector = AcneEnsembleDetector(api_key=api_key)

    # Run detection
    result, annotated = detector.detect(image_path, return_visualization=True)

    # Print results
    print("\n" + "=" * 50)
    print("DETECTION RESULTS")
    print("=" * 50)
    print(f"Acne count: {result.count}")
    print(f"Confidence: {result.confidence_level}")
    print(f"Primary model: {result.primary_model}")
    print(f"Image quality: {result.image_quality}")

    if result.classification_label:
        print(f"\nClassification: {result.classification_label} "
              f"({result.classification_confidence:.2f})")

    if result.warnings:
        print("\n⚠️ Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    # Assess severity
    severity_info = detector.assess_severity(result)
    print(f"\nSeverity: {severity_info['severity']}")
    print(f"Type breakdown: {severity_info['type_breakdown']}")

    # Save annotated image
    output_path = Path(image_path).stem + "_annotated.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"\n✓ Saved annotated image to {output_path}")
