from typing import List, Dict, Tuple
import io
import logging
from PIL import Image, ImageDraw
import json
import cv2
import numpy as np


logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(
        self,
        box_color: Tuple[int, int, int] = (255,0,0), # Red
        box_thickness: int = 3, 
        inpaint_radius: int = 5,
        inpaint_method: str = "NS"
    ):
        """
        Processes skin analysis images with bounding boxes and inpainting

        Features:
            - Draw bounding boxes around detected blemishes (before image)
            - Apply OpenCV inpainting to remove blemishes (after image)
            - Use normalized coordinates (0.0-1.0) for resolution independence
        """
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = inpaint_method

    def draw_bounding_boxes(self, image_bytes: bytes, analysis_json: str) -> bytes:
        """ This method will draw bounding boxes around detected blemishes
        
        Args:
            - image_bytes: This is the original image as bytes
            - analysis_json: JSON string from Gemini analysis

        Returns:
            Processed image with boxes as bytes
        """
        try: 
            # 1. Parse JSON and extract regions
            regions = self._extract_regions(analysis_json)
            
            # 2. Handle edge case that covers no blemishes detected
            if len(regions) == 0:
                logger.info("No blemishes detected. returning original image")
                return image_bytes
            
            # 3. Load image from bytes
            image_buffer = io.BytesIO(image_bytes)
            image = Image.open(image_buffer)
            
            # 4. Get image dimensions
            image_width, image_height = image.size

            # 5. Convert normalized coordinates to pixels
            pixel_regions = self._normalize_to_pixels(regions, image_width, image_height)

            # 6. Draw bounding boxes
            draw = ImageDraw.Draw(image) # Create the drawing context first
            for region in pixel_regions:  # Then draw each bounding box and use to draw a rectangle
                draw.rectangle(
                    ((region["x_min"], region["y_min"]), # Top-left point of the rectangle
                    (region["x_max"], region["y_max"])),  # Bottom-right point of the rectangle
                    outline=self.box_color, width=self.box_thickness
                )

            # 7. Convert image back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format="JPEG", quality=95)
            return output_buffer.getvalue()
        
        except Exception as e:
            logger.error(f"There was an error drawing bounding boxes: {e}")
            return image_bytes




    def apply_inpainting(self, image_bytes: bytes, analysis_json: str) -> bytes:
        """ This method is to remove blemishes using OpenCV inpainting

            Args:
                - image_bytes: This is the original image as bytes
                - analysis_json: JSON string from Gemini analysis
            
            Returns:
                Inpainted image as bytes (JPEG)
        """
        try:
            # 1. Extract regions
            regions = self._extract_regions(analysis_json)

            # 2. Handle edge case (no blemishes)
            if not regions:
                logger.warning(f"No blemishes detected. Returning original image")
                return image_bytes
            
            # 3. Load image from bytes
            # BUT Convert to Numpy array for OpenCV
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image_pil)

            # 4. Convert RGB (PIL) to BGR (OpenCV)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # 5. Get image dimensions
            image_height, image_width = image_bgr.shape[:2] # OpenCV uses (height, width)

            # 6. Convert normalized coordinates to pixels
            pixel_regions = self._normalize_to_pixels(regions, image_width, image_height)

            # 7. Create a mask (white = areas to inpaint)
            mask = np.zeros((image_height, image_width), dtype=np.uint8)  # Black image
            for region in pixel_regions:
                # Draw white rectangles on mask
                cv2.rectangle(
                    mask,
                    (region["x_min"], region["y_min"]),  # Top left
                    (region["x_max"], region["y_max"]),  # Bottom right
                    255,                                 # white color
                    -1                                   # fill the rectangle (not just outline it)
                )
            
            # 8. Apply inpainting
            inpainted_bgr = cv2.inpaint(
                src=image_bgr,
                inpaintMask=mask,
                inpaintRadius=self.inpaint_radius,
                inpaintMethod=cv2.INPAINT_TELEA if self.inpaint_method != "NS" else cv2.INPAINT_NS
            )

            # 9. Convert BGR back to RGB
            inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

            # 10. Convert NumPy array back to PIL Image
            inpainted_pil = Image.fromarray(inpainted_rgb)

            # 11. Convert PIL Image to bytes
            output_buffer = io.BytesIO()
            inpainted_pil.save(output_buffer, format="JPEG", quality=95)
            return output_buffer.getvalue()
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            return image_bytes
        except Exception as e:
            logger.error(f"There was an error applying inpainting: {e}")
            return image_bytes

    def _extract_regions(self, analysis_json: str) -> List[Dict]:
        """ Extract blemish areas from the Gemini JSON response

        Args:
            - analysis_json: JSON string from Gemini
        
        Returns:
            List of regions dicts with normalized coordinates e.g.:
            [
                {
                    "type": "papule",
                    "x_min": 0.2, "y_min": 0.3,
                    "x_max": 0.25, "y_max": 0.35,
                    "confidence":0.9
                }
            ]
        """
        try:
            # 1. Parse the JSON
            analysis_data = json.loads(analysis_json)

            # 2. Extract blemish_regions array
            blemish_regions = analysis_data.get("blemish_regions", [])

            # 3. Validate type: List
            if not isinstance(blemish_regions, list):
                logger.warning(f"blemish regions is not a list: {type(blemish_regions)}")
                return []
            
            # 4. Log the count of total blemish areas
            logger.info(f"Extracted {len(blemish_regions)} blemishes")

            return blemish_regions
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error extracting regions: {e}")
            return []


    def _normalize_to_pixels(self, regions: List[Dict],
        image_width: int, image_height: int
        ) -> List[Dict]:
        """
        Converts normalized coordinates (0.0-1.0) to pixel coordinates

        Args:
            - regions: List of dicts:
                List of regions e.g.
                [
                    {
                        "type": "papule",
                        "x_min": 0.2, "y_min": 0.3,
                        "x_max": 0.25, "y_max": 0.35,
                        "confidence":0.9
                    }
                ]
            - image_height: height of the original image as int
            - image_width: width of the original image as int

        Returns:
            List of dicts:
                List of regions with pixel coordinates
        """
        try:
            # 1. Extract the normalized regions
            pixel_regions = []
            for region in regions:
                x_min_norm = region.get("x_min", 0.0)
                y_min_norm = region.get("y_min", 0.0)
                x_max_norm = region.get("x_max", 0.0)
                y_max_norm = region.get("y_max", 0.0)

                # 2. Clamp to valid range [0.0-1.0]
                # AI might output 1.00001 or -0.0001 due to floating point
                x_min_norm = max(0.0, min(1.0, x_min_norm))
                y_min_norm = max(0.0, min(1.0, y_min_norm))
                x_max_norm = max(0.0, min(1.0, x_max_norm))
                y_max_norm = max(0.0, min(1.0, y_max_norm))

                # 3. Convert to pixel coordinates
                pixel_region = {
                    'type': region.get('type', 'general_blemish'),
                    'x_min': int(x_min_norm * image_width),
                    'y_min': int(y_min_norm * image_height),
                    'x_max': int(x_max_norm * image_width),
                    'y_max': int(y_max_norm * image_height),
                    'confidence': region.get('confidence', 0.0)
                }

                # 4. Validate that box has only positive dimensions
                if pixel_region["x_max"] <= pixel_region["x_min"]:
                    logger.warning(f"Invalid box width: {pixel_region}")
                    continue

                if pixel_region["y_max"] <= pixel_region["y_min"]:
                    logger.warning(f"Invalid box height: {pixel_region}")
                    continue

                pixel_regions.append(pixel_region)

            logger.info(f"Converted {len(pixel_regions)} regions to pixel coordinates")
            return pixel_regions
        except Exception as e:
            logger.error(f"Unexpected error converting to pixels: {e}")
            return []

