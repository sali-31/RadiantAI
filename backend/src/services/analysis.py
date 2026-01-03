import os
from dotenv import load_dotenv
import asyncio
import functools
import json
from typing import Dict, Any, List
import google.generativeai as genai
from google.cloud import vision
import logging

logger = logging.getLogger(__name__)

# Load environment variables (Generative AI)
load_dotenv()

#1. Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Configure Cloud Vision (Discriminative AI)
try:
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Google Cloud Vision failed to initialize. Check credentials and try again: {e}")
    vision_client = None

def _run_gemini_sync(image_bytes: bytes, mime_type: str) -> str:
    """
    Blocking call to Gemini 2.5 Flash with strict JSON output enforcement.

    This is the Brain of the ensemble.
    It performs qualitative analysis on the image, providing recommendations, characterizations, etc.

    Returns:
        JSON string with structured analysis
    """

    try:
        # 1. Set the response type to JSON
        generation_config = {
            "temperature": 0.4,  # Lower temperature reduces hallucinations
            "response_mime_type": "application/json"
        }

        # 2. Initialize model with JSON enforcement
        # Using Gemini 2.5 Flash for speed and efficiency
        model = genai.GenerativeModel(
            'gemini-2.5-flash', 
            generation_config=generation_config
        )

        # 3. Explicitly define the JSON schema in the prompt
        prompt = """
            You are an expert dermatological assistant.
            Analyze this image for skin conditions, specifically focusing on acne, blemishes, and lesions.

            Return the result ONLY as a valid JSON object matching this exact schema:
            {
                "characterization": "string description of the type of lesions/skin conditions observed",
                "detected_conditions": [
                    {
                        "condition": "string name of condition (e.g. Acne, Rosacea, Hyperpigmentation)",
                        "severity": "Mild" | "Moderate" | "Severe"
                    }
                ],
                "location": "string description of where the condition appears on the face/body",
                "recommendation": "string with general skincare advice including medical disclaimer",
                "treatments": ["array", "of", "recommended", "active", "ingredients"],
                "blemish_regions": [
                    {
                        "type": "papule" | "pustule" | "comedone" | "cyst" | "general_blemish",
                        "x_min": 0.0,
                        "y_min": 0.0,
                        "x_max": 1.0,
                        "y_max": 1.0,
                        "confidence": 0.0
                    }
                ]
            }

            Important:
            - Do not identify the person
            - Be specific about the skin condition type
            - Always include a medical disclaimer in recommendations
            - List 3-5 specific active ingredients that treat the identified condition
            - For blemish_regions: 
                - Use normalized coordinates (0.0 to 1.0) where (0,0) is top-left and (1,1) is bottom-right.
                - Ensure x_min < x_max and y_min < y_max.
                - Be precise with the bounding boxes.
            - Identify ALL visible blemishes, pustules, papules, or lesions
            - Set confidence between 0.0-1.0 based on visibility/certainty
            """

        response = model.generate_content([
            {"mime_type": mime_type, "data": image_bytes},
            prompt
        ])

        # Optional: Validate JSON parses correctly before returning
        try:
            json.loads(response.text)
        except json.JSONDecodeError as json_err:
            logger.warning(f"Warning: Gemini returned invalid JSON: {json_err}")
            # Fall through to return the text anyway, let the caller handle it

        return response.text

    except Exception as e:
        logger.error(f"Gemini API Error: {type(e).__name__}: {e}")
        # Return a valid fallback JSON so frontend doesn't crash
        return json.dumps({
            "characterization": "Analysis unavailable due to API error",
            "severity": "Unknown",
            "location": "Unknown",
            "recommendation": "Please try uploading the image again. If the problem persists, consult a dermatologist.",
            "treatments": [],
            "blemish_regions": []
        })
    
def _run_vision_sync(image_bytes: bytes) ->List[Dict[str, Any]]:
    """
    This is a blocking call to Google Cloud Vision
    This is the Eyes of the Ensemble;
    It detects objects/lables and returns bounding boxes.
    """

    if not vision_client:
        return []
    
    try:
        # Create the Vision API image object from bytes
        image = vision.Image(content=image_bytes)

        # Now we'll use object_localization to get bounding boxes
        objects = vision_client.object_localization(image=image).localized_object_annotations

        results = []
        for obj in objects:
            results.append({
                "name": obj.name,
                "confidence": obj.score,
                # Normalized vertices (0.0 to 1.0)
                "box": [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
            })
        return results
    except Exception as e:
        logger.error(f"There was an error with the Google Vision API: {e}")
        return []
    
async def perform_ensemble_analysis(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    This is the Orchestrator
    It runs both AI models in parallel using a ThreadPoolExecutor
    """
    loop = asyncio.get_running_loop()

    # Create partial functions to pass arguments to the sync functions
    gemini_func = functools.partial(_run_gemini_sync, image_bytes, mime_type)
    vision_func = functools.partial(_run_vision_sync, image_bytes)

    # Schedule both tasks to run immediately
    task1 = loop.run_in_executor(None, gemini_func)
    task2 = loop.run_in_executor(None, vision_func)

    # Wait for both to complete
    gemini_result, vision_result = await asyncio.gather(task1, task2)

    return {
        "analysis": gemini_result,
        "detections": vision_result
    }
