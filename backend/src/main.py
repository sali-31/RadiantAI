import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services.analysis import perform_ensemble_analysis
from .services.chatbot import SkinHealthChatbot
from .services.privacy import scrub_image_metadata
from .services.product_recommender import ProductRecommender


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ANALYSIS = {
    "characterization": (
        "The image was processed successfully. Cloud AI analysis is not configured "
        "for this local run, so RadiantAI is showing a general acne-prone skin routine."
    ),
    "detected_conditions": [{"condition": "Acne", "severity": "Moderate"}],
    "severity": "Moderate",
    "location": "Visible skin area",
    "recommendation": (
        "Use a gentle cleanser, introduce active treatments slowly, moisturize, and "
        "wear sunscreen daily. This is educational guidance, not a medical diagnosis."
    ),
    "treatments": ["salicylic acid", "benzoyl peroxide", "niacinamide"],
    "blemish_regions": [],
}

app = FastAPI(
    title="RadiantAI API",
    description="Privacy-preserving skin analysis and skincare recommendation API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

_recommender: Optional[ProductRecommender] = None
_chatbot: Optional[SkinHealthChatbot] = None


class RecommendRequest(BaseModel):
    analysis_text: str
    budget_max: Optional[float] = Field(default=None, ge=0)
    keywords: Optional[List[str]] = None


class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None


class ImageUrlRequest(BaseModel):
    s3_key: str


def get_recommender() -> ProductRecommender:
    global _recommender
    if _recommender is None:
        _recommender = ProductRecommender()
    return _recommender


def get_chatbot() -> Optional[SkinHealthChatbot]:
    global _chatbot
    if _chatbot is not None:
        return _chatbot
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    _chatbot = SkinHealthChatbot()
    return _chatbot


def public_upload_url(filename: str) -> str:
    return f"http://localhost:8000/uploads/{filename}"


def normalize_json(value: Any) -> Any:
    """Convert pandas/numpy values and NaN-like floats into JSON-safe objects."""
    if hasattr(value, "item"):
        return normalize_json(value.item())
    if isinstance(value, float) and (value != value):
        return None
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return value


def analysis_text_or_default(ai_analysis: Dict[str, Any]) -> str:
    analysis_text = ai_analysis.get("analysis")
    if not analysis_text:
        return json.dumps(DEFAULT_ANALYSIS)
    try:
        parsed = json.loads(analysis_text)
        if not parsed.get("detected_conditions"):
            parsed["detected_conditions"] = DEFAULT_ANALYSIS["detected_conditions"]
        return json.dumps(parsed)
    except (TypeError, json.JSONDecodeError):
        fallback = DEFAULT_ANALYSIS.copy()
        fallback["characterization"] = str(analysis_text)
        return json.dumps(fallback)


def infer_chat_condition(message: str) -> str:
    text = message.lower()
    condition_keywords = {
        "acne": ["acne", "pimple", "breakout", "blemish", "blackhead", "whitehead"],
        "rosacea": ["rosacea", "redness", "flushing"],
        "eczema": ["eczema", "itchy", "rash", "dermatitis"],
        "dry_skin": ["dry", "flaky", "dehydrated"],
        "oily_skin": ["oily", "oil", "shine", "sebum"],
        "hyperpigmentation": ["dark spot", "dark spots", "hyperpigmentation", "discoloration"],
        "sensitive_skin": ["sensitive", "irritation", "stinging", "burning"],
    }
    for condition, keywords in condition_keywords.items():
        if any(keyword in text for keyword in keywords):
            return condition
    return "acne"


def local_chat_response(message: str) -> Dict[str, Any]:
    text = message.lower()
    condition = infer_chat_condition(message)
    product_keywords: List[str] = []

    if any(word in text for word in ["routine", "regimen", "steps", "morning", "night"]):
        response = (
            "Here is a simple routine you can start with:\n\n"
            "- Morning: gentle cleanser, lightweight moisturizer, then SPF 30+.\n"
            "- Evening: cleanser, one targeted treatment, then moisturizer.\n"
            "- Introduce actives slowly, about 2-3 nights per week at first, so you can watch for irritation.\n\n"
            "If symptoms are painful, spreading, scarring, or not improving after several weeks, check in with a dermatologist."
        )
        product_keywords = ["cleanser", "treatment", "moisturizer"]
    elif any(word in text for word in ["benzoyl", "salicylic", "retinol", "adapalene", "niacinamide"]):
        response = (
            "Those are common acne-support ingredients, but they work differently:\n\n"
            "- Salicylic acid helps unclog pores and can be useful for blackheads or oily skin.\n"
            "- Benzoyl peroxide targets acne-causing bacteria and inflamed breakouts.\n"
            "- Adapalene supports cell turnover and is often useful for persistent acne.\n"
            "- Niacinamide can help calm redness and support the skin barrier.\n\n"
            "Pick one strong active at a time, moisturize well, and use sunscreen during the day."
        )
        product_keywords = ["salicylic", "benzoyl", "adapalene"]
    elif any(word in text for word in ["dry", "irritated", "barrier", "sensitive", "burning"]):
        response = (
            "For dryness or irritation, simplify for a few days:\n\n"
            "- Pause exfoliating acids, retinoids, and harsh scrubs.\n"
            "- Use a gentle cleanser or just rinse with water in the morning.\n"
            "- Apply a bland moisturizer with barrier-support ingredients like ceramides or glycerin.\n"
            "- Use sunscreen, especially if your skin is inflamed or healing.\n\n"
            "If burning, swelling, or a rash persists, it is worth getting medical advice."
        )
        product_keywords = ["moisturizer", "cream", "gentle"]
    elif any(word in text for word in ["dark", "spot", "hyperpigmentation", "marks", "scar"]):
        response = (
            "For dark spots or post-acne marks, consistency matters:\n\n"
            "- Daily sunscreen is the most important step, because UV exposure can keep marks darker.\n"
            "- Niacinamide, azelaic acid, vitamin C, or gentle retinoids may help brighten over time.\n"
            "- Avoid picking at breakouts, since that can prolong discoloration.\n\n"
            "Expect progress over weeks to months, not overnight."
        )
        product_keywords = ["sunscreen", "niacinamide", "vitamin"]
    else:
        response = (
            "I can help with skincare routines, ingredients, and product ideas. A good baseline is: "
            "cleanse gently, treat one concern at a time, moisturize, and use sunscreen every morning. "
            "Tell me your main concern, like acne, dryness, redness, oily skin, or dark spots, and I can tailor the advice."
        )

    products = []
    if product_keywords:
        products = get_recommender().recommend_for_condition(
            condition,
            budget_max=None,
            top_n=5,
            keywords=product_keywords,
        )
        if not products:
            products = get_recommender().recommend_for_condition(condition, top_n=5)

    return normalize_json({"response": response, "products": products[:5]})


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "RadiantAI API"}


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
    bundle_mode: bool = Form(default=True),
    budget_max: Optional[float] = Form(default=None),
) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        clean_bytes = scrub_image_metadata(raw_bytes)
    except Exception as exc:
        logger.exception("Image metadata scrub failed")
        raise HTTPException(status_code=400, detail="Could not process this image.") from exc

    filename = f"{uuid.uuid4().hex}.png"
    image_path = UPLOAD_DIR / filename
    image_path.write_bytes(clean_bytes)

    if os.getenv("GOOGLE_API_KEY"):
        ai_analysis = await perform_ensemble_analysis(clean_bytes, "image/png")
        ai_analysis["analysis"] = analysis_text_or_default(ai_analysis)
    else:
        ai_analysis = {"analysis": json.dumps(DEFAULT_ANALYSIS), "detections": []}

    recommender = get_recommender()
    if bundle_mode:
        product_recommendations = recommender.create_product_bundle_from_analysis(
            ai_analysis["analysis"],
            budget_max=budget_max,
        )
    else:
        product_recommendations = recommender.recommend_from_analysis(
            ai_analysis["analysis"],
            budget_max=budget_max,
        )

    return normalize_json(
        {
            "message": "Image analyzed successfully",
            "user_id": user_id,
            "s3_path": public_upload_url(filename),
            "s3_key": filename,
            "ai_analysis": ai_analysis,
            "product_recommendations": product_recommendations,
        }
    )


@app.post("/recommend")
def recommend_products(request: RecommendRequest) -> Dict[str, Any]:
    result = get_recommender().create_product_bundle_from_analysis(
        request.analysis_text,
        budget_max=request.budget_max,
        keywords=request.keywords,
    )
    return normalize_json(result)


@app.post("/api/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    chatbot = get_chatbot()
    if chatbot is None:
        return local_chat_response(request.message)

    try:
        response_data = chatbot.chat(request.message, request.conversation_history)
        product_names = response_data.get("recommended_products", [])
        products = get_recommender().find_products_by_names(product_names) if product_names else []
        return normalize_json(
            {
                "response": response_data.get("response_text", ""),
                "products": products[:5],
            }
        )
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail="Chat service is temporarily unavailable.") from exc


@app.post("/api/image-url")
def image_url(request: ImageUrlRequest) -> Dict[str, str]:
    image_path = UPLOAD_DIR / Path(request.s3_key).name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"url": public_upload_url(image_path.name)}
