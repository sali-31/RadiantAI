import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services.analysis import perform_ensemble_analysis
from .services.chatbot import SkinHealthChatbot
from .services.live_catalog import products_for_step, search_live_catalog, search_verified_catalog
from .services.memory import get_user_memory, memory_summary, update_user_memory
from .services.privacy import scrub_image_metadata
from .services.product_recommender import ProductRecommender
from .services.skincare_knowledge import (
    build_memory_updates,
    get_retrieved_knowledge,
    infer_chat_intent,
    infer_concerns,
    infer_skin_type,
)
from .services.skincare_nlu import SkincareNLU
from .services.skincare_rag import SkincareRAG
from .services.storage import is_s3_configured, looks_like_s3_key, presigned_s3_url, upload_image_to_s3


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

ROUTINE_STEPS = {
    "cleanser": ["cleanser", "cleansing", "face wash", "wash"],
    "toner": ["toner", "essence", "pad"],
    "treatment": [
        "treatment",
        "serum",
        "ampoule",
        "retinol",
        "adapalene",
        "benzoyl",
        "salicylic",
        "kojic",
        "tranexamic",
        "azelaic",
        "arbutin",
        "niacinamide",
        "snail",
    ],
    "moisturizer": ["moisturizer", "cream", "lotion", "barrier"],
    "sunscreen": ["sunscreen", "spf", "sun cream", "sun serum", "sunstick"],
}

STEP_CATEGORY_ALIASES = {
    "cleanser": ["cleanser"],
    "toner": ["toner", "essence"],
    "treatment": ["treatment", "serum", "ampoule", "spot_treatment"],
    "moisturizer": ["moisturizer", "cream", "lotion"],
    "sunscreen": ["sunscreen"],
}

PRODUCT_REQUEST_TERMS = [
    "recommend",
    "recommendation",
    "product",
    "products",
    "buy",
    "serum",
    "serums",
    "cleanser",
    "toner",
    "moisturizer",
    "cream",
    "sunscreen",
    "spf",
    "ampoule",
    "treatment",
]

ROUTINE_INTENT_TERMS = ["routine", "regimen", "steps", "morning", "night", "am", "pm"]

INGREDIENT_QUESTION_TERMS = ["ingredient", "ingredients", "active", "actives", "what should i use"]

CONCERN_TERMS = [
    "brightening",
    "dull",
    "dullness",
    "dark spot",
    "dark spots",
    "hyperpigmentation",
    "melasma",
    "anti aging",
    "anti-aging",
    "acne",
    "dry",
    "oily",
    "sensitive",
    "redness",
]

QUERY_STOPWORDS = {
    "good",
    "best",
    "give",
    "need",
    "want",
    "with",
    "from",
    "for",
    "the",
    "and",
    "products",
    "product",
    "recommend",
    "recommendations",
}

CURATED_MARKETPLACE_PRODUCTS = [
    {
        "title": "COSRX Low pH Good Morning Gel Cleanser",
        "brand": "COSRX",
        "category": "cleanser",
        "price": "See retailer",
        "rating": 4.7,
        "reviews": 12000,
        "retailer": "StyleKorean",
        "source": "StyleKorean",
        "link": "https://www.stylekorean.com/search/cosrx%20low%20ph%20good%20morning%20gel%20cleanser",
        "directions": "Use morning or evening on damp skin, massage gently, then rinse.",
        "reason": "Popular Korean low-pH cleanser for a gentle first step.",
    },
    {
        "title": "Anua Heartleaf Quercetinol Pore Deep Cleansing Foam",
        "brand": "Anua",
        "category": "cleanser",
        "price": "See retailer",
        "rating": 4.6,
        "reviews": 9000,
        "retailer": "StyleKorean",
        "source": "StyleKorean",
        "link": "https://www.stylekorean.com/search/anua%20heartleaf%20quercetinol%20pore%20deep%20cleansing%20foam",
        "directions": "Use as a gentle foaming cleanser, especially when skin feels oily or congested.",
        "reason": "Popular K-beauty cleanser from Anua for pores and oily skin.",
    },
    {
        "title": "LANEIGE Cream Skin Toner & Moisturizer",
        "brand": "LANEIGE",
        "category": "toner",
        "price": "See retailer",
        "rating": 4.6,
        "reviews": 5000,
        "retailer": "Sephora",
        "source": "Sephora",
        "link": "https://www.sephora.com/search?keyword=laneige%20cream%20skin%20toner%20moisturizer",
        "directions": "Apply after cleansing with palms or a cotton pad before serum.",
        "reason": "Popular Korean hydrating toner-moisturizer step at Sephora.",
    },
    {
        "title": "Anua Heartleaf 77 Soothing Toner",
        "brand": "Anua",
        "category": "toner",
        "price": "See retailer",
        "rating": 4.7,
        "reviews": 15000,
        "retailer": "StyleKorean",
        "source": "StyleKorean",
        "link": "https://www.stylekorean.com/search/anua%20heartleaf%2077%20soothing%20toner",
        "directions": "Pat into clean skin before treatment products.",
        "reason": "Popular Korean soothing toner for redness-prone or sensitive routines.",
    },
    {
        "title": "Beauty of Joseon Glow Serum Propolis + Niacinamide",
        "brand": "Beauty of Joseon",
        "category": "treatment",
        "price": "See retailer",
        "rating": 4.7,
        "reviews": 14000,
        "retailer": "StyleKorean",
        "source": "StyleKorean",
        "link": "https://www.stylekorean.com/search/beauty%20of%20joseon%20glow%20serum%20propolis%20niacinamide",
        "directions": "Use after toner and before moisturizer, starting once daily.",
        "reason": "Popular K-beauty serum for glow, oil balance, and post-blemish care.",
    },
    {
        "title": "Dr. Jart+ Cicapair Tiger Grass Serum",
        "brand": "Dr. Jart+",
        "category": "treatment",
        "price": "See retailer",
        "rating": 4.5,
        "reviews": 3000,
        "retailer": "Sephora",
        "source": "Sephora",
        "link": "https://www.sephora.com/search?keyword=dr%20jart%20cicapair%20tiger%20grass%20serum",
        "directions": "Apply after toner to help calm visible redness before moisturizer.",
        "reason": "Popular Korean cica-focused treatment available at Sephora.",
    },
    {
        "title": "Dr. Jart+ Ceramidin Skin Barrier Moisturizing Cream",
        "brand": "Dr. Jart+",
        "category": "moisturizer",
        "price": "See retailer",
        "rating": 4.6,
        "reviews": 7000,
        "retailer": "Sephora",
        "source": "Sephora",
        "link": "https://www.sephora.com/search?keyword=dr%20jart%20ceramidin%20cream",
        "directions": "Use as the final moisturizer step, morning or night.",
        "reason": "Popular Korean barrier cream for dryness or irritation.",
    },
    {
        "title": "LANEIGE Water Bank Blue Hyaluronic Cream Moisturizer",
        "brand": "LANEIGE",
        "category": "moisturizer",
        "price": "See retailer",
        "rating": 4.6,
        "reviews": 6000,
        "retailer": "Sephora",
        "source": "Sephora",
        "link": "https://www.sephora.com/search?keyword=laneige%20water%20bank%20blue%20hyaluronic%20cream",
        "directions": "Apply after serum to seal hydration.",
        "reason": "Popular Korean moisturizer for hydration-focused routines.",
    },
    {
        "title": "Beauty of Joseon Relief Sun Rice + Probiotics SPF50+",
        "brand": "Beauty of Joseon",
        "category": "sunscreen",
        "price": "See retailer",
        "rating": 4.8,
        "reviews": 20000,
        "retailer": "StyleKorean",
        "source": "StyleKorean",
        "link": "https://www.stylekorean.com/search/beauty%20of%20joseon%20relief%20sun",
        "directions": "Use every morning as the final skincare step and reapply when outdoors.",
        "reason": "Very popular K-beauty sunscreen for a comfortable daily SPF step.",
    },
    {
        "title": "innisfree Daily UV Defense Sunscreen SPF 36",
        "brand": "innisfree",
        "category": "sunscreen",
        "price": "See retailer",
        "rating": 4.5,
        "reviews": 9000,
        "retailer": "Sephora",
        "source": "Sephora",
        "link": "https://www.sephora.com/search?keyword=innisfree%20daily%20uv%20defense%20sunscreen",
        "directions": "Apply generously as the last morning step.",
        "reason": "Popular Korean SPF option available at Sephora.",
    },
]

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
_nlu: Optional[SkincareNLU] = None
_rag: Optional[SkincareRAG] = None


class RecommendRequest(BaseModel):
    analysis_text: str
    budget_max: Optional[float] = Field(default=None, ge=0)
    keywords: Optional[List[str]] = None


class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    skin_profile: Optional[Dict[str, Any]] = None


class ImageUrlRequest(BaseModel):
    s3_key: str


class LiveProductSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=12, ge=1, le=30)


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


def get_nlu() -> SkincareNLU:
    global _nlu
    if _nlu is None:
        _nlu = SkincareNLU()
    return _nlu


def get_rag() -> SkincareRAG:
    global _rag
    if _rag is None:
        _rag = SkincareRAG()
    return _rag


def public_upload_url(filename: str) -> str:
    base_url = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000").rstrip("/")
    return f"{base_url}/uploads/{filename}"


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
        "acne": [
            "acne",
            "pimple",
            "breakout",
            "blemish",
            "blackhead",
            "blackheads",
            "whitehead",
            "whiteheads",
            "comedone",
            "comedones",
            "closed comedone",
            "closed comedones",
            "clogged pore",
            "clogged pores",
            "tiny bumps",
            "forehead bumps",
        ],
        "rosacea": ["rosacea", "redness", "flushing"],
        "eczema": ["eczema", "itchy", "rash", "dermatitis"],
        "dry_skin": ["dry", "flaky", "dehydrated"],
        "oily_skin": ["oily", "oil", "shine", "sebum"],
        "hyperpigmentation": [
            "dark spot",
            "dark spots",
            "dark_spots",
            "hyperpigmentation",
            "discoloration",
            "melasma",
            "kojic",
            "brightening",
            "dullness",
            "tranexamic",
            "txa",
            "arbutin",
            "vitamin c",
        ],
        "sensitive_skin": ["sensitive", "irritation", "stinging", "burning"],
    }
    for condition, keywords in condition_keywords.items():
        if any(keyword in text for keyword in keywords):
            return condition
    return "general_skincare"


def condition_from_nlu(nlu: Dict[str, Any], fallback_message: str = "") -> str:
    concerns = nlu.get("concerns") or []
    concern_set = set(concerns)
    if {"closed_comedones", "acne"} & concern_set:
        return "acne"
    if {"dark_spots", "hyperpigmentation", "melasma", "brightening", "dullness"} & concern_set:
        return "hyperpigmentation"
    if "anti_aging" in concern_set:
        return "hyperpigmentation"
    if "dry_skin" in concern_set:
        return "dry_skin"
    if "oily_skin" in concern_set:
        return "oily_skin"
    if {"sensitive_skin", "redness", "barrier_repair"} & concern_set:
        return "sensitive_skin"
    fallback = infer_chat_condition(fallback_message)
    return fallback if fallback != "acne" else "general_skincare"


def wants_product_recommendations(message: str) -> bool:
    text = message.lower()
    is_ingredient_question = any(term in text for term in INGREDIENT_QUESTION_TERMS)
    explicit_product = any(term in text for term in PRODUCT_REQUEST_TERMS)
    routine = any(term in text for term in ROUTINE_INTENT_TERMS)

    if is_ingredient_question and not explicit_product:
        return False
    return explicit_product or routine


def is_routine_request(message: str) -> bool:
    text = message.lower()
    return any(term in text for term in ROUTINE_INTENT_TERMS)


def expanded_query_tokens(query: str) -> List[str]:
    tokens: List[str] = []
    for token in re.split(r"[^a-z0-9]+", query.lower()):
        if len(token) <= 2 or token in QUERY_STOPWORDS:
            continue
        tokens.append(token)
        if token.endswith("s") and len(token) > 4:
            tokens.append(token[:-1])
    return list(dict.fromkeys(tokens))


def requested_product_step(message: str) -> Optional[str]:
    text = message.lower()
    if any(term in text for term in ["sunscreen", "spf", "sun serum", "sun cream"]):
        return "sunscreen"
    for step, keywords in ROUTINE_STEPS.items():
        if any(keyword in text for keyword in keywords):
            return step
    return None


def is_serum_request(message: str) -> bool:
    return any(term in message.lower() for term in ["serum", "serums", "ampoule"])


def is_sunscreen_request(message: str) -> bool:
    return any(term in message.lower() for term in ["sunscreen", "spf", "sun cream", "sun serum"])


def filter_by_requested_form(message: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = message.lower()
    if is_serum_request(message):
        preferred = [
            product
            for product in products
            if any(
                form in f"{product.get('title', '')} {product.get('name', '')} {product.get('category', '')}".lower()
                for form in ["serum", "ampoule", "treatment"]
            )
        ]
        if len(preferred) >= 2:
            return preferred

    if is_sunscreen_request(message):
        preferred = [
            product
            for product in products
            if any(
                form in f"{product.get('title', '')} {product.get('name', '')} {product.get('category', '')}".lower()
                for form in ["sunscreen", "spf", "sun cream", "sun serum", "sunstick"]
            )
        ]
        if len(preferred) >= 2:
            return preferred

    form_terms = ["cleanser", "toner", "moisturizer", "cream"]
    requested_forms = [term.rstrip("s") for term in form_terms if term in text]
    if not requested_forms:
        return products

    filtered = [
        product
        for product in products
        if any(
            form in f"{product.get('title', '')} {product.get('name', '')} {product.get('category', '')}".lower()
            for form in requested_forms
        )
    ]
    return filtered if len(filtered) >= 2 else products


def related_conditions(condition: str) -> List[str]:
    if condition == "hyperpigmentation":
        return ["hyperpigmentation", "dark_spots", "melasma"]
    return [condition]


def search_local_product_catalog(query: str, condition: str, limit: int = 8) -> List[Dict[str, Any]]:
    catalog_df = get_recommender().get_combined_products(related_conditions(condition))
    if catalog_df.empty:
        return []

    tokens = expanded_query_tokens(query)
    if not tokens:
        return []

    df = catalog_df.copy()

    def score_row(row: Any) -> int:
        title = str(row.get("title", "")).lower()
        category = str(row.get("category", "")).lower()
        haystack = f"{title} {category}"
        score = 0
        for token in tokens:
            token_weight = 10 if token in {"kojic", "tranexamic", "txa", "arbutin", "azelaic"} else 4
            if token in title:
                score += token_weight
            elif token in haystack:
                score += max(2, token_weight // 2)
        if any(step_word in title for step_word in ["serum", "treatment", "ampoule"]):
            score += 2
        return score

    df["_query_score"] = df.apply(score_row, axis=1)
    df = df[df["_query_score"] > 0]
    if df.empty:
        return []

    sort_columns = ["_query_score"]
    ascending = [False]
    for column in ["rating", "reviews"]:
        if column in df.columns:
            sort_columns.append(column)
            ascending.append(False)

    return df.sort_values(sort_columns, ascending=ascending).head(limit).drop(columns=["_query_score"]).to_dict("records")


def products_for_chat_message(message: str, condition: str, limit: int = 12) -> List[Dict[str, Any]]:
    step = None if is_routine_request(message) else requested_product_step(message)
    verified_products = filter_by_requested_form(
        message,
        search_verified_catalog(message, step=step, limit=limit),
    )
    live_products = filter_by_requested_form(message, try_live_product_search(message, limit=8))
    local_products = filter_by_requested_form(
        message,
        search_local_product_catalog(message, condition, limit=8),
    )
    query_products = merge_products(
        verified_products,
        live_products,
        local_products,
        limit=limit,
    )

    if is_routine_request(message):
        routine_products = merge_products(
            build_step_recommendations(condition, query=message),
            query_products,
            limit=max(limit, 16),
        )
        return balanced_routine_products(routine_products, limit=min(limit, 10))

    if len(query_products) >= 2:
        return query_products

    return merge_products(
        query_products,
        build_step_recommendations(condition, query=message),
        limit=limit,
    )


def balanced_routine_products(products: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    steps = ["cleanser", "toner", "treatment", "moisturizer", "sunscreen"]
    by_step = {step: [] for step in steps}
    extras = []
    for product in products:
        step = product.get("routine_step") or product.get("category")
        if step in by_step:
            by_step[step].append(product)
        else:
            extras.append(product)

    balanced = []
    for round_index in range(2):
        for step in steps:
            if len(balanced) >= limit:
                return balanced
            if len(by_step[step]) > round_index:
                balanced.append(by_step[step][round_index])

    for step in steps:
        for product in by_step[step][2:]:
            if len(balanced) >= limit:
                return balanced
            balanced.append(product)

    for product in extras:
        if len(balanced) >= limit:
            return balanced
        balanced.append(product)

    return balanced


def align_response_with_product_count(response_text: str, products: List[Dict[str, Any]]) -> str:
    if len(products) <= 1:
        return response_text
    replacements = {
        "Here's a great option to consider:": "Here are some good options to consider:",
        "Here’s a great option to consider:": "Here are some good options to consider:",
        "Here is a great option to consider:": "Here are some good options to consider:",
    }
    aligned = response_text
    for original, replacement in replacements.items():
        aligned = aligned.replace(original, replacement)
    return aligned


def plain_chat_response_text(value: Any) -> str:
    text = str(value or "").strip()
    for _ in range(3):
        if not text.startswith("{"):
            return text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if '\\"response_text\\"' in text or "\\n" in text:
                unescaped = text.replace('\\"', '"').replace("\\n", "\n")
                try:
                    parsed = json.loads(unescaped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    next_text = parsed.get("response_text") or parsed.get("response")
                    if next_text:
                        text = str(next_text).strip()
                        continue
            match = re.search(
                r'\\?"response_text\\?"\s*:\s*\\?"(?P<text>.*?)(?<!\\)\\?"\s*(?:,|\})',
                text,
                flags=re.DOTALL,
            )
            if not match:
                return text
            text = match.group("text").replace('\\"', '"').replace("\\n", "\n").strip()
            continue
        if not isinstance(parsed, dict):
            return text
        next_text = parsed.get("response_text") or parsed.get("response")
        if not next_text or str(next_text).strip() == text:
            return text
        text = str(next_text).strip()
    return text


def append_follow_up_questions(response_text: str, questions: List[str]) -> str:
    clean_questions = [question.strip() for question in questions if question and question.strip()]
    if not clean_questions:
        return response_text
    if "follow-up" in response_text.lower() or "follow up" in response_text.lower():
        return response_text
    bullets = "\n".join(f"- {question}" for question in clean_questions[:3])
    return f"{response_text.rstrip()}\n\n**A few questions so I can fine-tune this:**\n{bullets}"


def product_names_from_context(products: List[Dict[str, Any]]) -> List[str]:
    names = []
    for product in products:
        name = product.get("name") or product.get("title")
        if name:
            names.append(str(name))
    return names


def merge_memory_updates(
    primary: Optional[Dict[str, Any]],
    secondary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    primary = primary or {}
    secondary = secondary or {}
    merged = {
        "skin_type": primary.get("skin_type") or secondary.get("skin_type"),
        "concerns": [],
        "goals": [],
        "preferences": [],
        "avoid": [],
    }
    for key in ["concerns", "goals", "preferences", "avoid"]:
        values: List[str] = []
        for source in [primary, secondary]:
            for value in source.get(key) or []:
                clean = str(value).strip()
                if clean and clean not in values and clean != "general_skincare":
                    values.append(clean)
        merged[key] = values
    return merged


def product_query_for_message(message: str, concerns: List[str], intent: Dict[str, bool]) -> str:
    concern_text = " ".join(concerns).replace("_", " ")
    if intent.get("serum_request"):
        if {"brightening", "dullness", "dark_spots", "hyperpigmentation"} & set(concerns):
            return "brightening dark spot serum vitamin c niacinamide tranexamic azelaic"
        if "acne" in concerns:
            return "acne serum salicylic niacinamide azelaic"
        return f"{concern_text} serum".strip() or message

    if intent.get("sunscreen_request"):
        return f"{concern_text} sunscreen spf".strip() or "sunscreen spf"

    if intent.get("routine_request"):
        return f"{concern_text} skincare routine cleanser toner serum moisturizer sunscreen".strip()

    return message


def product_query_from_nlu(message: str, nlu: Dict[str, Any]) -> str:
    concerns = [concern for concern in (nlu.get("concerns") or []) if concern != "general_skincare"]
    concern_text = " ".join(concerns).replace("_", " ")
    product_form = nlu.get("product_form")
    intent = nlu.get("intent")

    if product_form == "serum":
        if {"brightening", "dullness", "dark_spots", "hyperpigmentation", "melasma"} & set(concerns):
            return "brightening dark spot serum vitamin c niacinamide tranexamic azelaic"
        if {"closed_comedones", "acne"} & set(concerns):
            return "acne serum salicylic niacinamide azelaic"
        return f"{concern_text} serum".strip() or "serum"

    if product_form == "sunscreen":
        return f"{concern_text} sunscreen spf".strip() or "sunscreen spf"

    if product_form in {"cleanser", "toner", "moisturizer", "treatment"}:
        return f"{concern_text} {product_form}".strip() or product_form

    if intent == "routine_request" or product_form == "routine_bundle":
        return f"{concern_text} skincare routine cleanser toner serum moisturizer sunscreen".strip()

    return f"{concern_text} {message}".strip()


def local_chat_response(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    skin_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = message.lower()
    current_concerns = infer_concerns(message)
    concerns = current_concerns or infer_concerns(message, conversation_history)
    intent = infer_chat_intent(message)
    skin_type = infer_skin_type(message, conversation_history) or (skin_profile or {}).get("skin_type")
    condition = infer_chat_condition(" ".join(concerns) or message)
    wants_products = wants_product_recommendations(message)
    memory_updates = build_memory_updates(message, conversation_history)

    if any(term in text for term in ["wait between", "how long should i wait", "between skincare steps", "between steps", "layer skincare"]):
        response = (
            "You usually **do not need long wait times** between skincare steps.\n\n"
            "- For most steps, wait about **30-60 seconds**, or until the product no longer feels very wet.\n"
            "- Apply from thinnest to thickest: cleanser, toner/essence, serum, moisturizer, then sunscreen in the morning.\n"
            "- Give **sunscreen** a few minutes to set before makeup or going outside.\n"
            "- Wait longer only if a prescription label tells you to, if benzoyl peroxide needs to dry before clothing, or if layers are pilling.\n\n"
            "The bigger goal is even application and comfort, not a strict timer."
        )
        wants_products = False
    elif intent["ingredient_question"] and {"brightening", "dullness", "dark_spots", "hyperpigmentation"} & set(concerns + ["brightening" if "brighten" in text else ""]):
        response = (
            "For **brightening and uneven tone**, the most useful skincare ingredients are:\n\n"
            "- **Vitamin C:** best in the morning for antioxidant support, glow, and uneven tone.\n"
            "- **Niacinamide:** beginner-friendly; supports the barrier, dullness, redness, and oil balance.\n"
            "- **Tranexamic acid:** great for dark spots, hyperpigmentation, and melasma-style uneven pigment.\n"
            "- **Alpha arbutin:** pigment-targeting and usually gentle.\n"
            "- **Azelaic acid:** helpful if you have dark spots plus acne-prone skin or redness.\n"
            "- **AHAs like lactic or glycolic acid:** help surface dullness and texture; use 1-3 nights per week.\n"
            "- **Retinoids:** help texture, post-acne marks, and long-term tone; night only and introduce slowly.\n"
            "- **Sunscreen:** non-negotiable every morning, because brightening work stalls if UV keeps pigment active."
        )
    elif wants_products and intent["serum_request"] and {"brightening", "dullness", "dark_spots", "hyperpigmentation"} & set(concerns + ["brightening" if "brighten" in text else ""]):
        response = (
            "For **brightening serums**, choose the serum type based on the problem:\n\n"
            "- **Vitamin C serum:** best for glow, dullness, and morning antioxidant support.\n"
            "- **Tranexamic acid or alpha arbutin serum:** better for dark spots and uneven pigment.\n"
            "- **Azelaic acid serum:** good if you also get acne, redness, or post-breakout marks.\n"
            "- **Niacinamide serum:** the easiest beginner option for dullness, oil balance, redness, and barrier support.\n\n"
            "Use one brightening serum at a time, then moisturizer. Keep sunscreen daily."
        )
    elif not intent["routine_request"] and (
        "dark_spots" in concerns or "hyperpigmentation" in concerns or "dark spot" in text or "dark spots" in text
    ):
        response = (
            "For **dark spots and hyperpigmentation**, the goal is to slow new pigment and fade existing marks gradually.\n\n"
            "- **Sunscreen every morning** is the anchor step; UV exposure keeps dark spots active.\n"
            "- **Tranexamic acid** and **alpha arbutin** are strong pigment-focused options.\n"
            "- **Azelaic acid** is useful when dark spots come with acne or redness.\n"
            "- **Niacinamide** is a gentle supporting active for uneven tone and barrier health.\n"
            "- A slow-start **retinoid** can help post-acne marks and texture over time.\n\n"
            "Use one main treatment at a time and expect progress over weeks to months."
        )
    elif not intent["routine_request"] and ("dullness" in concerns or "skin concern is dull skin" in text):
        response = (
            "Got it — **dull skin** usually benefits from hydration, gentle exfoliation, and steady sun protection.\n\n"
            "- Start with a gentle cleanser so your skin does not feel stripped.\n"
            "- Add **niacinamide** or **vitamin C** for glow and uneven tone.\n"
            "- Use a gentle AHA, like lactic acid, **1-3 nights per week** if your skin tolerates exfoliation.\n"
            "- Keep a moisturizer in the routine so brightening ingredients do not dry you out.\n"
            "- Use **sunscreen every morning**; dullness and uneven tone improve much faster when UV exposure is controlled."
        )
    elif intent["routine_request"] and ("anti_aging" in concerns or "anti aging" in text or "anti-aging" in text):
        response = (
            "Here is an **anti-aging routine** that keeps the focus on prevention, texture, and barrier support:\n\n"
            "**Morning**\n"
            "1. Gentle cleanser or rinse.\n"
            "2. Optional **vitamin C** for antioxidant support and glow.\n"
            "3. Moisturizer with barrier-support ingredients.\n"
            "4. **Broad-spectrum sunscreen** every day.\n\n"
            "**Night**\n"
            "1. Cleanser.\n"
            "2. **Retinoid** 2-3 nights per week to start, then increase slowly.\n"
            "3. Moisturizer; peptides are optional if you want extra firming support.\n\n"
            "Go slowly with retinoids, especially if your skin is dry or sensitive."
        )
    elif intent["routine_request"] and {"dark_spots", "hyperpigmentation", "brightening", "dullness"} & set(concerns):
        response = (
            "Here is a **dark-spot/brightening routine**:\n\n"
            "**Morning**\n"
            "1. Gentle cleanser.\n"
            "2. Brightening serum: **vitamin C**, **niacinamide**, or **tranexamic acid**.\n"
            "3. Lightweight moisturizer.\n"
            "4. **Sunscreen SPF 30+** every morning; this is the most important step for dark spots.\n\n"
            "**Night**\n"
            "1. Cleanser.\n"
            "2. Treatment: **azelaic acid**, **alpha arbutin**, **tranexamic acid**, or a slow-start retinoid.\n"
            "3. Moisturizer to protect the barrier.\n\n"
            "Avoid stacking too many actives on the same night. Progress usually takes weeks to months."
        )
    elif intent["routine_request"] and "acne" in concerns:
        acne_intro = (
            "Here is a **closed-comedone routine** for clogged pores and tiny under-the-skin bumps:\n\n"
            if any(term in text for term in ["closed comedone", "closed comedones", "comedone", "comedones", "tiny bumps", "clogged pore"])
            else "Here is an **acne-focused routine** that treats breakouts without stripping the skin:\n\n"
        )
        response = (
            acne_intro +
            "**Morning**\n"
            "1. Gentle cleanser.\n"
            "2. Optional **niacinamide** to support oil balance and redness.\n"
            "3. Lightweight non-comedogenic moisturizer.\n"
            "4. **Sunscreen SPF 30+** every morning.\n\n"
            "**Night**\n"
            "1. Cleanser.\n"
            "2. Treatment: **salicylic acid** for clogged pores/closed comedones, or **adapalene** for persistent comedonal acne. Use benzoyl peroxide mainly if bumps become inflamed pimples.\n"
            "3. Moisturizer to reduce dryness and irritation.\n\n"
            "Start with one active at a time. If acne is painful, scarring, or spreading, check in with a dermatologist."
        )
    elif intent["routine_request"] and {"dry_skin", "sensitive_skin", "barrier_repair", "redness"} & set(concerns):
        response = (
            "Here is a **dry/sensitive barrier-support routine**:\n\n"
            "**Morning**\n"
            "1. Rinse or use a very gentle cleanser.\n"
            "2. Hydrating layer with **glycerin** or **hyaluronic acid** if tolerated.\n"
            "3. Moisturizer with **ceramides** or barrier-support ingredients.\n"
            "4. Gentle **sunscreen** every morning.\n\n"
            "**Night**\n"
            "1. Gentle cleanser.\n"
            "2. Skip acids/retinoids while stinging or burning is active.\n"
            "3. Rich moisturizer; petrolatum can help seal very dry areas.\n\n"
            "Once your barrier feels calm, reintroduce actives slowly."
        )
    elif "acne" in concerns:
        response = (
            "For **acne**, match the active to the breakout type:\n\n"
            "- **Salicylic acid:** clogged pores, blackheads, oily skin.\n"
            "- **Benzoyl peroxide:** inflamed pimples.\n"
            "- **Adapalene/retinoids:** persistent acne and post-acne marks.\n"
            "- Use a non-comedogenic moisturizer and sunscreen so treatment does not wreck your barrier.\n\n"
            "If acne is painful, scarring, or not improving after several weeks, a dermatologist can help."
        )
    elif {"dry_skin", "sensitive_skin", "barrier_repair", "redness"} & set(concerns):
        response = (
            "For **dry, sensitive, or barrier-stressed skin**, simplify first:\n\n"
            "- Use a gentle cleanser or just rinse in the morning.\n"
            "- Look for **ceramides, glycerin, hyaluronic acid**, and, when needed, petrolatum-based occlusives.\n"
            "- Pause exfoliating acids and retinoids if your skin stings or burns.\n"
            "- Rebuild with moisturizer and sunscreen before adding strong treatments back."
        )
    else:
        response = (
            "A good baseline is: cleanse gently, treat one concern at a time, moisturize, and use sunscreen every morning. "
            "If you share your skin type, main concern, and whether you want ingredients, a routine, or product recommendations, I can tailor it."
        )

    product_query = product_query_for_message(message, concerns, intent)
    products = products_for_chat_message(product_query, condition, limit=12) if wants_products else []
    if wants_products:
        products = merge_products(
            products,
            search_local_product_catalog(product_query, condition, limit=8),
            limit=12,
        )

    return normalize_json(
        {
            "response": align_response_with_product_count(plain_chat_response_text(response), products),
            "products": products,
            "memory_updates": memory_updates,
        }
    )


def smart_local_chat_response(
    message: str,
    user_id: str,
    memory: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    skin_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = local_chat_response(message, conversation_history=conversation_history, skin_profile=skin_profile)
    products = response.get("products", [])
    updated_memory = update_user_memory(user_id, message, products)
    response["memory"] = updated_memory
    response["memory_updates"] = response.get("memory_updates") or build_memory_updates(message, conversation_history)
    return response


def build_step_recommendations(condition: str, per_step: int = 2, query: str = "") -> List[Dict[str, Any]]:
    recommender = get_recommender()
    catalog_df = recommender.get_combined_products([condition])
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for step, keywords in ROUTINE_STEPS.items():
        step_products = [
            product.copy()
            for product in CURATED_MARKETPLACE_PRODUCTS
            if product["category"] == step
        ][:1]

        for product in step_products:
            product["routine_step"] = step
            seen.add(product_identity(product))
            selected.append(product)

        for product in products_for_step(step, query=query or condition, limit=per_step):
            if len([item for item in selected if item.get("routine_step") == step]) >= per_step:
                break
            product["routine_step"] = step
            product["category"] = step
            identity = product_identity(product)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(product)

        if not catalog_df.empty and len(step_products) < per_step:
            df = catalog_df.copy()
            title_matches = df["title"].str.contains("|".join(keywords), case=False, na=False)
            if "category" in df.columns:
                category_values = df["category"].astype(str).str.lower()
                category_matches = category_values.isin(STEP_CATEGORY_ALIASES[step])
                known_step_categories = {
                    category
                    for aliases in STEP_CATEGORY_ALIASES.values()
                    for category in aliases
                }
                non_conflicting_title_match = ~category_values.isin(known_step_categories)
            else:
                category_matches = title_matches & False
                non_conflicting_title_match = title_matches
            matches = df[category_matches | (title_matches & non_conflicting_title_match)].copy()

            if not matches.empty:
                matches = matches.sort_values(["rating", "reviews"], ascending=[False, False])
                for product in matches.to_dict("records"):
                    product["category"] = step
                    product["routine_step"] = step
                    identity = product_identity(product)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    selected.append(product)
                    if len([item for item in selected if item.get("routine_step") == step]) >= per_step:
                        break

        if len([item for item in selected if item.get("routine_step") == step]) < per_step:
            for product in CURATED_MARKETPLACE_PRODUCTS:
                if product["category"] != step:
                    continue
                identity = product_identity(product)
                if identity in seen:
                    continue
                fallback_product = product.copy()
                fallback_product["routine_step"] = step
                seen.add(identity)
                selected.append(fallback_product)
                if len([item for item in selected if item.get("routine_step") == step]) >= per_step:
                    break

    return selected


def merge_products(*product_groups: List[Dict[str, Any]], limit: int = 16) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for products in product_groups:
        for product in products:
            identity = product_identity(product)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(product)
            if len(merged) >= limit:
                return merged
    return merged


def try_live_product_search(query: str, limit: int = 12) -> List[Dict[str, Any]]:
    if os.getenv("ENABLE_LIVE_PRODUCT_SEARCH", "true").lower() in {"0", "false", "no"}:
        return []
    try:
        return search_live_catalog(query, limit=limit)
    except Exception as exc:
        logger.info("Live product search unavailable: %s", exc)
        return []


def product_identity(product: Dict[str, Any]) -> str:
    return "|".join(
        str(product.get(key, "")).lower()
        for key in ["retailer", "brand", "asin", "title", "name", "link"]
    )


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
    storage_result = {
        "provider": "local",
        "key": filename,
        "url": public_upload_url(filename),
    }

    if is_s3_configured():
        try:
            storage_result = upload_image_to_s3(clean_bytes, filename, "image/png")
        except RuntimeError as exc:
            logger.warning("S3 upload failed; falling back to local upload storage: %s", exc)
            image_path = UPLOAD_DIR / filename
            image_path.write_bytes(clean_bytes)
    else:
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
            "s3_path": storage_result["url"],
            "s3_key": storage_result["key"],
            "storage_provider": storage_result["provider"],
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

    user_id = request.user_id or "anonymous"
    memory = get_user_memory(user_id)
    nlu = get_nlu().classify(
        request.message,
        conversation_history=request.conversation_history,
        skin_profile=request.skin_profile,
    )
    concerns = nlu.get("concerns") or ["general_skincare"]
    retrieved_context = get_rag().context_text(request.message, nlu, top_k=5)
    product_query = product_query_from_nlu(request.message, nlu)
    condition = condition_from_nlu(nlu, product_query)
    should_preload_products = bool(nlu.get("needs_products", False))
    retrieved_products = products_for_chat_message(product_query, condition, limit=12) if should_preload_products else []

    chatbot = get_chatbot()
    if chatbot is None:
        return smart_local_chat_response(
            request.message,
            user_id,
            memory,
            conversation_history=request.conversation_history,
            skin_profile=request.skin_profile,
        )

    try:
        response_data = chatbot.chat(
            request.message,
            request.conversation_history,
            skin_profile=request.skin_profile,
            nlu=nlu,
            retrieved_context=retrieved_context,
            product_context=retrieved_products,
            memory_context=memory_summary(memory),
        )
        product_names = response_data.get("recommended_products", [])
        product_query = response_data.get("product_query") or product_query
        model_wants_products = bool(response_data.get("wants_products", False))
        should_show_products = model_wants_products or bool(nlu.get("needs_products", False))

        products = get_recommender().find_products_by_names(product_names) if product_names else []
        if should_show_products:
            condition = condition_from_nlu(nlu, product_query)
            products = merge_products(
                products,
                products_for_chat_message(product_query, condition, limit=12),
                limit=12,
            )
        if should_show_products and not response_data.get("recommended_products"):
            response_data["recommended_products"] = product_names_from_context(products)

        updated_memory = update_user_memory(user_id, request.message, products)
        memory_updates = response_data.get("memory_updates") or build_memory_updates(
            request.message,
            request.conversation_history,
        )
        memory_updates = merge_memory_updates(memory_updates, nlu.get("memory_updates"))
        response_text = append_follow_up_questions(
            plain_chat_response_text(response_data.get("response_text", "")),
            response_data.get("followup_questions", []),
        )
        return normalize_json(
            {
                "response": align_response_with_product_count(response_text, products),
                "products": products,
                "memory": updated_memory,
                "memory_updates": memory_updates,
                "nlu": nlu,
            }
        )
    except Exception as exc:
        logger.warning("Gemini chat failed; using local skincare fallback: %s", exc)
        return smart_local_chat_response(
            request.message,
            user_id,
            memory,
            conversation_history=request.conversation_history,
            skin_profile=request.skin_profile,
        )


@app.post("/api/image-url")
def image_url(request: ImageUrlRequest) -> Dict[str, str]:
    if is_s3_configured() and looks_like_s3_key(request.s3_key):
        url = presigned_s3_url(request.s3_key)
        if not url:
            raise HTTPException(status_code=404, detail="Image not found.")
        return {"url": url}

    image_path = UPLOAD_DIR / Path(request.s3_key).name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return {"url": public_upload_url(image_path.name)}


@app.post("/api/live-products/search")
def live_product_search(request: LiveProductSearchRequest) -> Dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    live_products = try_live_product_search(request.query, limit=request.limit)
    verified_products = search_verified_catalog(request.query, limit=request.limit)
    products = merge_products(live_products, verified_products, limit=request.limit)
    return normalize_json(
        {
            "query": request.query,
            "live_count": len(live_products),
            "verified_count": len(verified_products),
            "products": products,
        }
    )
