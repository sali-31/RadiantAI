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
    if any(term in text for term in ["no longer want product", "stop recommending products", "just explain", "do not recommend products", "don't recommend products"]):
        return False
    is_ingredient_question = any(term in text for term in INGREDIENT_QUESTION_TERMS)
    explicit_product = any(
        term in text
        for term in [
            "recommend",
            "recommendation",
            "product",
            "products",
            "buy",
            "affordable",
            "under $",
            "only serums",
            "only sunscreens",
            "only moisturizers",
            "only cleansers",
            "complete routine",
        ]
    )
    product_form_question = any(
        term in text
        for term in [
            "what cleanser should i use",
            "what moisturizer should i use",
            "what sunscreen should i use",
            "what serum should i use",
            "what sunscreen is best",
            "what cleanser is best",
            "what moisturizer is best",
            "best cleanser",
            "best sunscreen",
            "best moisturizer",
        ]
    )
    informational_product_question = any(
        term in text
        for term in [
            "how much sunscreen",
            "how often should i reapply",
            "do i need sunscreen",
            "can i skip sunscreen",
            "is spf",
            "does tinted sunscreen",
            "mineral vs chemical",
        ]
    )
    routine = any(term in text for term in ROUTINE_INTENT_TERMS if term not in {"am", "pm"}) or bool(
        re.search(r"\b(?:am|pm)\b", text)
    )
    safety = any(
        term in text
        for term in [
            "rash",
            "swollen",
            "pus",
            "painful",
            "spreading",
            "blister",
            "scarring",
            "stop using",
            "should i pop",
        ]
    )

    if safety:
        return False

    if is_ingredient_question and not explicit_product:
        return False
    if informational_product_question:
        return False
    return explicit_product or product_form_question or routine


def is_routine_request(message: str) -> bool:
    text = message.lower()
    return any(term in text for term in ROUTINE_INTENT_TERMS if term not in {"am", "pm"}) or bool(
        re.search(r"\b(?:am|pm)\b", text)
    )


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
    if any(term in text for term in ["sunscreen", "sunscreens", "spf", "sun serum", "sun cream"]):
        return "sunscreen"
    if "cleansers" in text:
        return "cleanser"
    if "moisturizers" in text:
        return "moisturizer"
    for step, keywords in ROUTINE_STEPS.items():
        if any(keyword in text for keyword in keywords):
            return step
    return None


def is_serum_request(message: str) -> bool:
    return any(term in message.lower() for term in ["serum", "serums", "ampoule"])


def is_sunscreen_request(message: str) -> bool:
    return any(term in message.lower() for term in ["sunscreen", "sunscreens", "spf", "sun cream", "sun serum"])


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


def filter_products_by_avoid_terms(products: List[Dict[str, Any]], avoid_terms: List[str]) -> List[Dict[str, Any]]:
    clean_terms = [term for term in avoid_terms if term and len(term) >= 3]
    if not clean_terms:
        return products
    filtered = []
    for product in products:
        text = " ".join(
            str(product.get(field, ""))
            for field in ["title", "name", "brand", "category", "routine_step", "directions", "reason"]
        ).lower()
        if any(term in text for term in clean_terms):
            continue
        filtered.append(product)
    return filtered or products


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


def avoid_terms_from_profile(skin_profile: Optional[Dict[str, Any]], memory_updates: Dict[str, Any]) -> List[str]:
    avoid: List[str] = []
    for source in [skin_profile or {}, memory_updates or {}]:
        for key in ["avoid", "avoided_ingredients"]:
            for value in source.get(key) or []:
                clean = str(value).lower().strip()
                if clean and clean not in avoid:
                    avoid.append(clean)
    return avoid


def message_says_avoid(text: str, ingredient: str) -> bool:
    if ingredient not in text:
        return False
    return any(
        phrase in text
        for phrase in [
            "cannot tolerate",
            "cannot use",
            "can't tolerate",
            "can't use",
            "cant tolerate",
            "cant use",
            "can not tolerate",
            "can not use",
            "allergic",
            "sensitive to",
            "react to",
            "reacts badly",
            "react badly",
            "breaks me out",
            "avoid",
            "without",
            f"no {ingredient}",
        ]
    )


def local_chat_response(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    skin_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = message.lower()
    current_concerns = infer_concerns(message)
    intent = infer_chat_intent(message)
    use_history_context = intent.get("followup") and not any(
        term in text
        for term in [
            "rash",
            "swollen",
            "painful",
            "pus",
            "blister",
            "white patches",
            "stop using",
            "wait between",
            "how long should i wait",
        ]
    )
    concerns = current_concerns or (infer_concerns(message, conversation_history) if use_history_context else [])
    skin_type = infer_skin_type(message, conversation_history) or (skin_profile or {}).get("skin_type")
    condition = infer_chat_condition(" ".join(concerns) or message)
    wants_products = wants_product_recommendations(message)
    memory_updates = build_memory_updates(message, conversation_history)
    avoid_terms = avoid_terms_from_profile(skin_profile, memory_updates)
    avoid_niacinamide = message_says_avoid(text, "niacinamide") or "niacinamide" in avoid_terms
    avoid_vitamin_c = message_says_avoid(text, "vitamin c") or "vitamin c" in avoid_terms
    avoid_retinol = message_says_avoid(text, "retinol") or message_says_avoid(text, "retinoid") or any(term in avoid_terms for term in ["retinol", "retinoid"])
    avoid_salicylic = message_says_avoid(text, "salicylic acid") or "salicylic acid" in avoid_terms
    avoid_benzoyl = message_says_avoid(text, "benzoyl peroxide") or "benzoyl peroxide" in avoid_terms
    avoid_hyaluronic = message_says_avoid(text, "hyaluronic acid") or "hyaluronic acid" in avoid_terms
    avoid_fragrance = message_says_avoid(text, "fragrance") or "fragrance" in avoid_terms
    avoid_exfoliating_acids = "no exfoliating acid" in text or "without exfoliating acid" in text or "exfoliating acids" in avoid_terms

    if intent.get("safety_or_medical") or any(
        term in text
        for term in [
            "rash",
            "spreading",
            "painful",
            "swollen",
            "pus",
            "red, hot",
            "hot and swollen",
            "blister",
            "blistering",
            "scarring",
            "white patches",
            "should i pop",
            "stop using",
            "face burn",
            "burns",
            "itchy",
            "cracked",
        ]
    ):
        response = (
            "This sounds like a situation where **skincare alone may not be enough**.\n\n"
            "- **Stop** any new or irritating product for now.\n"
            "- Keep the area gentle and **barrier-focused**: mild cleanser, plain moisturizer, and sunscreen if the skin is intact.\n"
            "- Once calm, **patch test** new products before using them all over, especially if your skin is sensitive.\n"
            "- Avoid exfoliating acids, retinoids, vitamin C, scrubs, or popping/squeezing inflamed bumps.\n"
            "- If symptoms are **painful, spreading, hot, swollen, pus-filled, blistering, scarring, or persistent**, contact a **doctor or dermatologist**. Seek urgent care for facial swelling, rapidly spreading redness/heat, or a painful one-sided blistering rash."
        )
        wants_products = False
    elif avoid_retinol and any(term in text for term in ["anti-aging", "anti aging", "wrinkle", "fine line", "retinol"]):
        response = (
            "If your skin reacts badly to **retinol**, you can still build an anti-aging routine without it.\n\n"
            "- **Sunscreen SPF 30+ daily:** the strongest anti-aging step for prevention.\n"
            "- **Peptides:** optional support for firmness and barrier comfort.\n"
            "- **Vitamin C:** antioxidant support and brightness if you tolerate it.\n"
            "- **Niacinamide:** barrier, tone, and fine-line support if you tolerate it.\n"
            "- **Azelaic acid:** helpful for tone, redness, and post-breakout marks with less retinoid-style irritation.\n"
            "- **Gentle AHAs/PHAs:** can improve texture, but use slowly and skip them if your barrier is irritated.\n\n"
            "Keep the routine boring around those actives: gentle cleanser, moisturizer, and sunscreen."
        )
        wants_products = wants_product_recommendations(message)
    elif avoid_hyaluronic and any(term in text for term in ["hydrating", "hydration", "hydrate", "hyaluronic"]):
        response = (
            "If **hyaluronic acid** does not work for you, use other hydration and barrier ingredients instead:\n\n"
            "- **Glycerin:** excellent, common humectant for water-binding hydration.\n"
            "- **Panthenol:** soothing hydration and barrier support.\n"
            "- **Beta-glucan:** calming, cushiony hydration for sensitive skin.\n"
            "- **Aloe or centella:** soothing hydration if your skin tolerates botanicals.\n"
            "- **Ceramides and squalane:** help reduce water loss so skin stays comfortable longer.\n\n"
            "Apply moisturizer on slightly damp skin and avoid over-layering watery serums if they make you feel tight."
        )
        wants_products = wants_product_recommendations(message)
    elif avoid_salicylic and any(term in text for term in ["blackhead", "blackheads", "clogged pore", "closed comedone"]):
        response = (
            "If you cannot use **salicylic acid**, you still have options for blackheads and clogged pores:\n\n"
            "- **Adapalene:** helps prevent new clogged pores over time; start 2-3 nights per week.\n"
            "- **Gentle oil cleansing:** can help loosen sunscreen and excess sebum before a regular cleanser.\n"
            "- **Clay mask occasionally:** can reduce surface oil, but avoid drying your skin out.\n"
            "- **Non-comedogenic moisturizer and sunscreen:** prevent irritation that can worsen congestion.\n\n"
            "Avoid harsh scrubs and pore strips as your main strategy; they can irritate without fixing the clog cycle."
        )
        wants_products = wants_product_recommendations(message)
    elif avoid_benzoyl and intent["routine_request"] and "acne" in concerns:
        response = (
            "Here is an **acne routine without benzoyl peroxide**:\n\n"
            "**Morning**\n"
            "1. Gentle cleanser.\n"
            "2. Lightweight, non-comedogenic moisturizer.\n"
            "3. Sunscreen SPF 30+.\n\n"
            "**Night**\n"
            "1. Cleanser.\n"
            "2. Choose one: **salicylic acid** for clogged pores/blackheads, **azelaic acid** for acne plus redness or marks, or **adapalene** for persistent comedones.\n"
            "3. Moisturizer to reduce irritation.\n\n"
            "Introduce one active at a time. If acne is painful, scarring, or not improving, a dermatologist can help."
        )
    elif any(term in text for term in ["essential oils", "essential oil"]):
        response = (
            "If your skin hates **essential oils**, avoid products that use them as fragrance or sensorial ingredients.\n\n"
            "Common label terms to watch for include lavender oil, tea tree oil, peppermint oil, citrus oils, eucalyptus oil, rosemary oil, limonene, linalool, citral, geraniol, and general **fragrance/parfum**.\n\n"
            "Choose fragrance-free products, patch test new formulas, and be extra cautious with leave-on products like serums, moisturizers, and sunscreen."
        )
        wants_products = False
    elif avoid_niacinamide and any(
        term in text
        for term in ["brighten", "brightening", "glow", "dark spot", "dark spots", "hyperpigmentation", "dull"]
    ):
        exfoliation_line = (
            "- Skip exfoliating acids for this request; lean on pigment-targeting non-acid options instead.\n"
            if avoid_exfoliating_acids
            else "- **Lactic acid or glycolic acid:** useful for dull surface texture, but start only **1-2 nights per week**.\n"
        )
        response = (
            "Yes - **skip niacinamide** if you know your skin does not tolerate it.\n\n"
            "For brightening without relying on niacinamide, look at these options:\n\n"
            "- **Vitamin C:** best for glow and antioxidant support, especially in the morning if your skin tolerates it.\n"
            "- **Tranexamic acid:** a strong choice for dark spots, hyperpigmentation, and uneven pigment.\n"
            "- **Alpha arbutin:** a gentle pigment-focused option for uneven tone.\n"
            "- **Azelaic acid:** helpful for dark spots, post-acne marks, redness, and acne-prone skin.\n"
            f"{exfoliation_line}"
            "- **Sunscreen SPF 30+:** the non-negotiable step, because brightening results fade if UV keeps triggering pigment.\n\n"
            "If your skin is sensitive, I would start with azelaic acid or alpha arbutin, then add stronger actives slowly."
        )
        wants_products = wants_product_recommendations(message)
    elif any(term in text for term in ["wait between", "how long should i wait", "between skincare steps", "between steps", "layer skincare"]):
        response = (
            "You usually **do not need long wait times** between skincare steps.\n\n"
            "- For most steps, wait about **30-60 seconds**, or until the product no longer feels very wet.\n"
            "- Apply from thinnest to thickest: cleanser, toner/essence, serum, moisturizer, then sunscreen in the morning.\n"
            "- Give **sunscreen** a few minutes to set before makeup or going outside.\n"
            "- Wait longer only if a prescription label tells you to, if benzoyl peroxide needs to dry before clothing, or if layers are pilling.\n\n"
            "The bigger goal is even application and comfort, not a strict timer."
        )
        wants_products = False
    elif any(term in text for term in ["pie and pih", "pie vs pih", "pih acne", "pie acne"]):
        response = (
            "**PIE and PIH are different types of post-acne marks:**\n\n"
            "- **PIE** means post-inflammatory erythema. It looks red, pink, or purple and comes from visible blood-vessel inflammation after acne.\n"
            "- **PIH** means post-inflammatory hyperpigmentation. It looks brown, gray-brown, or darker than your skin tone and comes from excess melanin.\n"
            "- Both need **daily sunscreen** because UV can make marks last longer.\n"
            "- PIE often benefits from calming care, barrier repair, azelaic acid, and time.\n"
            "- PIH often benefits from tranexamic acid, azelaic acid, alpha arbutin, retinoids if tolerated, and consistent sunscreen.\n\n"
            "If marks are pitted or indented, that is scarring rather than color change, and skincare has limits."
        )
        wants_products = False
    elif any(term in text for term in ["back acne", "chest acne", "body wash helps acne"]):
        response = (
            "For **body acne**, use acne actives without over-drying the skin:\n\n"
            "- Try a **benzoyl peroxide wash** in the shower for inflamed bumps; rinse well because it can bleach towels.\n"
            "- Use **salicylic acid body wash** for clogged pores and oiliness.\n"
            "- Shower after sweating and change out of tight sweaty clothing.\n"
            "- Use a lightweight moisturizer if the skin gets dry or itchy.\n"
            "- If acne is painful, scarring, or widespread, a dermatologist can help."
        )
        wants_products = wants_product_recommendations(message)
    elif any(term in text for term in ["redness"]) and any(term in text for term in ["ingredient", "ingredients", "explain"]):
        response = (
            "For **redness-prone skin**, choose calming and barrier-supporting ingredients before strong actives:\n\n"
            "- **Azelaic acid:** useful for redness, acne-prone skin, and post-breakout marks if tolerated.\n"
            "- **Centella asiatica:** soothing support for irritation-prone skin.\n"
            "- **Panthenol and glycerin:** hydration and barrier comfort.\n"
            "- **Ceramides:** help repair the skin barrier so redness triggers less easily.\n"
            "- **Mineral or gentle sunscreen:** UV can worsen redness and sensitivity.\n\n"
            "Avoid fragrance, essential oils, harsh scrubs, and stacking acids while the skin is reactive."
        )
        wants_products = False
    elif any(term in text for term in ["vitamin c and azelaic", "alpha arbutin with vitamin c", "tranexamic acid with retinol", "retinol and benzoyl peroxide", "salicylic acid and glycolic", "ingredients should not", "rotate retinol"]):
        response = (
            "For **ingredient mixing**, separate strong or irritating actives until you know your skin tolerates them.\n\n"
            "- **Vitamin C + azelaic acid:** often okay, but separate AM/PM if sensitive.\n"
            "- **Alpha arbutin + vitamin C:** usually compatible for brightening.\n"
            "- **Tranexamic acid + retinol:** can be used, but alternate or separate if dry or irritated.\n"
            "- **Retinol + benzoyl peroxide:** can be drying; separate them unless a formula is designed to combine them.\n"
            "- **Salicylic acid + glycolic acid:** avoid stacking at first; alternate nights to reduce over-exfoliation.\n\n"
            "A simple rotation: retinoid night, recovery night, exfoliating acid night, recovery night, brightening serum as tolerated."
        )
        wants_products = False
    elif any(term in text for term in ["retinol and glycolic", "vitamin c and niacinamide", "start using retinol", "retinol safely", "retinol every night"]):
        if "vitamin c and niacinamide" in text:
            response = (
                "Yes, **vitamin C and niacinamide** can usually be used **together** if your skin can tolerate the combination.\n\n"
                "- Use vitamin C in the morning if you want antioxidant support.\n"
                "- Niacinamide can support barrier, oil balance, redness, and uneven tone.\n"
                "- If you sting or flush easily, separate them AM/PM or alternate days.\n"
                "- Keep sunscreen daily, especially when working on tone."
            )
        elif "retinol and glycolic" in text:
            response = (
                "I would not start **retinol and glycolic acid** on the same night.\n\n"
                "- Use them on **alternate nights** to lower irritation risk.\n"
                "- Retinol supports acne, texture, and fine lines; glycolic acid exfoliates surface texture.\n"
                "- Add moisturizer before or after retinol if your skin is dry or sensitive.\n"
                "- If burning, peeling, or tightness shows up, pause actives and repair the barrier."
            )
        else:
            response = (
                "To start **retinol** safely, go slowly:\n\n"
                "- Use it at **night** only.\n"
                "- Start 2 nights per week, then increase slowly if your skin stays calm.\n"
                "- Use a **pea-sized** amount for the whole face.\n"
                "- Moisturizer can go before and after if you need buffering.\n"
                "- Wear sunscreen every morning because retinoids can increase sun sensitivity."
            )
        wants_products = False
    elif not wants_products and "sunscreen" in text and any(term in text for term in ["how much", "how often", "reapply", "indoors", "spf 30", "skip", "moisturizer has spf", "tinted", "melasma", "dark spots"]):
        response = (
            "For **sunscreen**, think amount, coverage, and reapplication:\n\n"
            "- Use enough for face and neck; the common guide is about **two finger lengths** for face/neck.\n"
            "- Reapply every **2 hours** when outdoors, sweating, or swimming.\n"
            "- Indoors, sunscreen still matters if you sit near bright windows or are treating dark spots, melasma, or anti-aging.\n"
            "- **SPF 30+ broad-spectrum** is a good daily minimum; SPF 50 can be helpful for long outdoor exposure.\n"
            "- Tinted sunscreen with **iron oxides** can help melasma/dark spots because visible light can worsen pigment.\n"
            "- Moisturizer with SPF only works if you apply enough; many people under-apply it."
        )
        wants_products = False
    elif any(term in text for term in ["order do i apply", "what order", "morning vs night", "morning and night", "damp or dry", "serum on damp", "moisturizer go before", "after retinol", "sunscreen right after moisturizer", "steps are actually necessary", "toner necessary", "essence and serum", "wash my face in the morning", "simplify my routine"]):
        response = (
            "For **routine application**, keep it practical:\n\n"
            "- Basic order: **cleanser**, toner/essence if you use one, **serum** or treatment, **moisturizer**, then **sunscreen** in the morning.\n"
            "- Morning-friendly ingredients include vitamin C, hydrating serums, and sunscreen.\n"
            "- Night-friendly ingredients include retinoids and exfoliating acids, used slowly.\n"
            "- Hydrating serums can go on slightly damp skin; strong actives are often better on dry skin to reduce irritation.\n"
            "- Moisturizer can go before retinol if you need buffering, or after retinol if your skin tolerates it.\n"
            "- Sunscreen goes after moisturizer and is the last morning step.\n"
            "- Toner, essence, and extra serums are optional, not required.\n"
            "- If irritated, simplify to gentle cleanser, moisturizer, sunscreen, and pause actives."
        )
        wants_products = False
    elif any(term in text for term in ["same as whiteheads", "salicylic acid or adapalene", "moisturizer cause closed comedones", "clear closed comedones", "avoid if i get clogged pores", "oil cleansing help closed comedones", "ingredients help clogged pores"]):
        response = (
            "**Closed comedones** are clogged pores under the skin; whiteheads are a type of closed comedone when the clog is more visible near the surface.\n\n"
            "- **Salicylic acid/BHA** helps inside oily pores and can be good for blackheads or mild clogs.\n"
            "- **Adapalene** is often stronger for persistent closed comedones because it helps normalize how pores shed.\n"
            "- Moisturizers can contribute if they are too heavy for your skin, but skipping moisturizer can also backfire by causing irritation.\n"
            "- Avoid heavy oils, waxy balms, thick occlusive layers all over acne-prone areas, and harsh scrubs.\n"
            "- Oil cleansing can help remove sunscreen/makeup, but follow with a gentle cleanser and stop if bumps worsen.\n\n"
            "Expect several weeks to a few months, especially with adapalene."
        )
        wants_products = False
    elif any(term in text for term in ["niacinamide do", "what does niacinamide", "ingredients help oily skin"]):
        response = (
            "**Niacinamide** is a flexible support ingredient:\n\n"
            "- Helps strengthen the **barrier** and reduce water loss.\n"
            "- Can help balance **oil** and shine.\n"
            "- May calm **redness** and uneven tone.\n"
            "- Can support a brighter, more even-looking **tone** over time.\n\n"
            "For oily skin, pair niacinamide with a gentle cleanser, lightweight moisturizer, and sunscreen; salicylic acid can help if clogged pores are also present."
        )
        wants_products = False
    elif any(term in text for term in ["retinol vs retinal", "retinal vs adapalene", "retinol, retinal", "retinol retinal adapalene"]):
        response = (
            "**Retinol, retinal, and adapalene** are all retinoid-family options, but they differ in strength and irritation risk:\n\n"
            "- **Retinol:** gentler, slower, good beginner anti-aging option.\n"
            "- **Retinal:** closer to active retinoic acid, often stronger/faster than retinol.\n"
            "- **Adapalene:** acne-focused retinoid, useful for comedones and breakouts.\n"
            "- Stronger does not always mean better; irritation can ruin consistency.\n\n"
            "Use at night, start slowly, moisturize, and wear sunscreen."
        )
        wants_products = False
    elif any(term in text for term in ["pregnant", "pregnancy"]):
        response = (
            "If you are **pregnant**, ask your doctor about your routine, especially prescription or high-strength actives.\n\n"
            "- Generally **avoid retinoids** unless your clinician specifically approves.\n"
            "- Often-discussed options include gentle cleanser, moisturizer, mineral sunscreen, and sometimes azelaic acid.\n"
            "- Avoid starting aggressive peels or strong multi-active routines.\n"
            "- Sunscreen is important for melasma-prone pigment changes during pregnancy."
        )
        wants_products = False
    elif any(term in text for term in ["dark spots on my body", "body dark spots"]):
        response = (
            "For **dark spots on the body**, combine pigment control with irritation control:\n\n"
            "- Use **sunscreen** on exposed areas so spots do not keep darkening.\n"
            "- Gentle **exfoliation** with lactic acid or glycolic acid can help roughness and surface dullness if tolerated.\n"
            "- Use moisturizer to reduce friction and barrier irritation.\n"
            "- Azelaic acid, tranexamic acid, or alpha arbutin can help uneven pigment.\n"
            "- If spots are rapidly changing, painful, or unexplained, check with a dermatologist."
        )
        wants_products = False
    elif any(term in text for term in ["dark circles"]):
        response = (
            "**Dark circles** can come from pigment, shadows, veins, allergies, sleep, or genetics, so skincare has limits.\n\n"
            "- Sunscreen helps prevent pigment from worsening.\n"
            "- Caffeine can temporarily reduce puffiness.\n"
            "- A gentle eye retinoid can help texture over time, but introduce slowly.\n"
            "- Sleep, allergies, and irritation matter too.\n\n"
            "Avoid strong acids close to the eyes."
        )
        wants_products = False
    elif any(term in text for term in ["large pores", "rough skin texture", "skin texture is rough"]):
        response = (
            "For **pores and rough texture**, focus on consistency rather than scrubbing:\n\n"
            "- **Niacinamide** can help the look of pores and oil balance.\n"
            "- **Salicylic acid** helps oily clogged pores.\n"
            "- **AHA** exfoliation can smooth rough surface texture.\n"
            "- **Retinoids** help texture over time.\n"
            "- Moisturizer and sunscreen keep the barrier healthy so texture treatments are tolerable."
        )
        wants_products = False
    elif (intent["ingredient_question"] or "what can i use" in text) and {"brightening", "dullness", "dark_spots", "hyperpigmentation"} & set(concerns + ["brightening" if "brighten" in text else ""]):
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
    elif not wants_products and not intent["routine_request"] and (
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
        glow_active = "**vitamin C**" if avoid_niacinamide else "**niacinamide** or **vitamin C**"
        response = (
            "Got it - **dull skin** usually benefits from hydration, gentle exfoliation, and steady sun protection.\n\n"
            "- Start with a gentle cleanser so your skin does not feel stripped.\n"
            f"- Add {glow_active} for glow and uneven tone.\n"
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
        morning_brighteners = (
            "**vitamin C**, **tranexamic acid**, **alpha arbutin**, or **azelaic acid**"
            if avoid_niacinamide
            else "**vitamin C**, **niacinamide**, or **tranexamic acid**"
        )
        if avoid_vitamin_c and avoid_niacinamide:
            morning_brighteners = "**tranexamic acid**, **alpha arbutin**, or **azelaic acid**"
        elif avoid_vitamin_c:
            morning_brighteners = "**tranexamic acid**, **alpha arbutin**, **azelaic acid**, or **niacinamide**"
        response = (
            "Here is a **dark-spot/brightening routine**:\n\n"
            "**Morning**\n"
            "1. Gentle cleanser.\n"
            f"2. Brightening serum: {morning_brighteners}.\n"
            "3. Lightweight moisturizer.\n"
            "4. **Sunscreen SPF 30+** every morning; this is the most important step for dark spots.\n\n"
            "**Evening/Night**\n"
            "1. Cleanser.\n"
            "2. Treatment: **azelaic acid**, **alpha arbutin**, **tranexamic acid**, or a slow-start retinoid.\n"
            "3. Moisturizer to protect the barrier.\n\n"
            "**Frequency and why:** use pigment treatments most days only if tolerated, exfoliate no more than 1-3 nights weekly, and keep sunscreen daily because UV keeps pigment active. Progress usually takes weeks to months."
        )
    elif intent["routine_request"] and "acne" in concerns:
        oil_balance_step = (
            "Optional hydrating serum if your skin feels tight."
            if avoid_niacinamide
            else "Optional **niacinamide** to support oil balance and redness."
        )
        acne_intro = (
            "Here is a **closed-comedone routine** for clogged pores and tiny under-the-skin bumps:\n\n"
            if any(term in text for term in ["closed comedone", "closed comedones", "comedone", "comedones", "tiny bumps", "clogged pore"])
            else "Here is an **acne-focused routine** that treats breakouts without stripping the skin:\n\n"
        )
        response = (
            acne_intro +
            "**Morning**\n"
            "1. Gentle cleanser.\n"
            f"2. {oil_balance_step}\n"
            "3. Lightweight non-comedogenic moisturizer.\n"
            "4. **Sunscreen SPF 30+** every morning.\n\n"
            "**Night**\n"
            "1. Cleanser.\n"
            "2. Treatment: **salicylic acid** for clogged pores/closed comedones, or **adapalene** for persistent comedonal acne. Use benzoyl peroxide mainly if bumps become inflamed pimples.\n"
            "3. Moisturizer to reduce dryness and irritation.\n\n"
            "Start with one active at a time. If acne is painful, scarring, or spreading, check in with a dermatologist."
        )
    elif intent["routine_request"] and "oily_skin" in concerns:
        response = (
            "Here is an **oily-skin routine** that controls shine without drying you out:\n\n"
            "**Morning**\n"
            "1. Gentle cleanser, not a stripping one.\n"
            "2. Optional **niacinamide** for oil balance if tolerated.\n"
            "3. **Lightweight moisturizer** or gel-cream.\n"
            "4. Lightweight sunscreen SPF 30+.\n\n"
            "**Night**\n"
            "1. Cleanser.\n"
            "2. Salicylic acid 2-3 nights weekly if clogged pores are present.\n"
            "3. Lightweight moisturizer.\n\n"
            "Keep moisturizer in the routine; dehydrated oily skin can feel even oilier."
        )
    elif intent["routine_request"] and {"dry_skin", "sensitive_skin", "barrier_repair", "redness"} & set(concerns):
        response = (
            "Here is a **dry/sensitive barrier-support routine**:\n\n"
            "**Morning**\n"
            "1. Rinse or use a very gentle **hydrating cleanser**.\n"
            "2. Hydrating layer with **glycerin** or **hyaluronic acid** if tolerated.\n"
            "3. Moisturizer with **ceramides** or barrier-support ingredients.\n"
            "4. Gentle **sunscreen** every morning.\n\n"
            "**Night**\n"
            "1. Gentle cleanser.\n"
            "2. Skip acids/retinoids while stinging or burning is active.\n"
            "3. Rich moisturizer; petrolatum or another **occlusive** can help seal very dry, **flaky** areas.\n\n"
            "Once your barrier feels calm, reintroduce actives slowly."
        )
    elif not intent["routine_request"] and any(term in text for term in ["blackhead", "blackheads"]):
        response = (
            "**Blackheads** are open clogged pores, so the goal is to dissolve oil inside the pore without scrubbing your skin raw.\n\n"
            "- Use a **salicylic acid/BHA leave-on** 2-3 nights per week to start.\n"
            "- Keep the rest gentle: mild cleanser, lightweight non-comedogenic moisturizer, and sunscreen.\n"
            "- Avoid harsh scrubs or squeezing; they can irritate the pore and make marks worse.\n"
            "- If blackheads are stubborn, a slow-start **adapalene/retinoid** at night can help prevent new clogs.\n\n"
            "Do not start BHA and a retinoid on the same night at first; alternate until your skin adjusts."
        )
        wants_products = wants_product_recommendations(message)
    elif not intent["routine_request"] and any(term in text for term in ["whitehead", "whiteheads", "closed comedone", "closed comedones"]):
        response = (
            "**Whiteheads** are closed comedones, meaning oil and dead skin are trapped under the pore opening.\n\n"
            "- Start with **salicylic acid/BHA** 2-3 nights per week for clogged pores.\n"
            "- If they keep coming back, consider **adapalene** at night because retinoids help prevent comedones from forming.\n"
            "- Use **benzoyl peroxide** mainly when bumps are red or inflamed, not as the only treatment for closed clogs.\n"
            "- Keep a gentle cleanser, non-comedogenic moisturizer, and sunscreen so treatment does not damage your barrier.\n\n"
            "Give comedonal acne several weeks; it usually improves gradually, not overnight."
        )
        wants_products = wants_product_recommendations(message)
    elif not intent["routine_request"] and any(term in text for term in ["tiny bumps", "forehead bumps", "bumps on my forehead"]):
        response = (
            "**Tiny bumps on the forehead** are often closed comedones, but if they are very itchy and all look the same, folliculitis can also be possible.\n\n"
            "- For clogged pores, try **salicylic acid/BHA** a few nights weekly or a slow-start **adapalene** routine.\n"
            "- Avoid heavy hair oils, pomades, and thick leave-in products near the hairline.\n"
            "- Keep cleanser and moisturizer gentle so you are treating clogs without causing irritation.\n"
            "- If the bumps are itchy, spreading, painful, or not improving, a dermatologist can check whether it is acne or something else.\n\n"
            "The key is to treat it like clogged pores first, but not force stronger acne products if the pattern does not fit."
        )
        wants_products = wants_product_recommendations(message)
    elif "acne" in concerns:
        response = (
            "For **acne**, match the active to the breakout type:\n\n"
            "- **Salicylic acid:** clogged pores, blackheads, oily skin.\n"
            "- **Benzoyl peroxide:** inflamed pimples.\n"
            "- **Adapalene/retinoids:** persistent acne and post-acne marks.\n"
            "- Use a non-comedogenic moisturizer and sunscreen so treatment does not wreck your barrier.\n\n"
            "If acne is painful, scarring, or not improving after several weeks, a dermatologist can help."
        )
    elif any(term in text for term in ["oily t-zone", "t-zone", "dry cheeks", "combination skin"]):
        response = (
            "That sounds like **combination skin**: oily **T-zone** with **dry cheeks**.\n\n"
            "- Use a gentle cleanser so the cheeks do not get stripped.\n"
            "- Use a **lightweight** moisturizer all over, then add a little extra moisturizer only on dry cheeks.\n"
            "- If the T-zone clogs, use salicylic acid only there 1-3 nights per week.\n"
            "- Choose a lightweight sunscreen that does not feel greasy.\n\n"
            "Treat zones differently instead of using the same strong product on every area."
        )
    elif not wants_products and {"dry_skin", "sensitive_skin", "barrier_repair", "redness"} & set(concerns):
        response = (
            "For **dry, sensitive, or barrier-stressed skin**, simplify first:\n\n"
            "- Use a gentle cleanser or just rinse in the morning.\n"
            "- Look for **ceramides, glycerin, hyaluronic acid**, and, when needed, petrolatum-based occlusives.\n"
            "- Pause exfoliating acids and retinoids if your skin stings or burns.\n"
            "- Rebuild with moisturizer and sunscreen before adding strong treatments back."
        )
    elif "difference between" in text:
        response = (
            "Here is the practical difference:\n\n"
            "- **Dry vs dehydrated:** dry skin lacks oil; dehydrated skin lacks water and can still feel oily but tight.\n"
            "- **PIE vs PIH:** PIE is red/pink post-acne marking from blood vessels; PIH is brown/gray pigment from melanin.\n"
            "- **Blackheads vs sebaceous filaments:** blackheads are clogged pores; sebaceous filaments are normal oil structures that refill.\n"
            "- **Retinol, retinal, tretinoin:** retinol is gentler, retinal is stronger/faster, tretinoin is prescription-strength retinoic acid.\n"
            "- **Mineral vs chemical sunscreen:** mineral filters sit more on top and can be better around sensitive eyes; chemical filters are often lighter.\n"
            "- **Humectants, emollients, occlusives:** humectants hydrate, emollients soften, occlusives seal water in.\n"
            "- **AHA, BHA, PHA:** AHA smooths surface texture, BHA helps oily clogged pores, PHA is gentler.\n"
            "- **Purging vs irritation:** purging happens where you normally break out after acne actives; irritation brings burning, rashy redness, or new areas.\n\n"
            "If you tell me which pair you mean, I can go deeper on that exact comparison."
        )
        wants_products = False
    elif any(term in text for term in ["back acne", "chest acne", "body wash helps acne"]):
        response = (
            "For **body acne**, use acne actives without over-drying the skin:\n\n"
            "- Try a **benzoyl peroxide wash** in the shower for inflamed bumps; rinse well because it can bleach towels.\n"
            "- Use **salicylic acid body wash** for clogged pores and oiliness.\n"
            "- Shower after sweating and change out of tight sweaty clothing.\n"
            "- Use a lightweight moisturizer if the skin gets dry or itchy.\n"
            "- If acne is painful, scarring, or widespread, a dermatologist can help."
        )
    elif any(term in text for term in ["rough bumps on arms", "dark underarms", "dark knees", "dark elbows", "ingrown hairs", "razor bumps", "flaky scalp", "dandruff", "hairline"]):
        response = (
            "For **body, shaving, and scalp concerns**, match the approach to the pattern:\n\n"
            "- Rough bumps on arms often behave like keratosis pilaris: use lactic acid, urea, or gentle moisturizing regularly.\n"
            "- Ingrown hairs/razor bumps improve with gentle exfoliation, shaving with the grain, fewer blade passes, and moisturizer.\n"
            "- Dark underarms, knees, or elbows often involve friction plus pigment; reduce irritation, moisturize, use sunscreen on exposed areas, and consider azelaic acid or gentle lactic acid if tolerated.\n"
            "- Dandruff/flaky scalp often responds to shampoos with ketoconazole, zinc pyrithione, or selenium sulfide.\n"
            "- Hairline pimples can come from hair products; keep oils/pomades off the hairline and cleanse after sweating."
        )
        wants_products = wants_product_recommendations(message)
    elif any(term in text for term in ["skincare pill", "vitamin c serum turn", "product is expired", "patch test", "non-comedogenic", "comedogenic ratings", "alcohol always bad", "natural ingredients safer", "diet affect acne", "stress and sleep"]):
        response = (
            "Here is the practical answer:\n\n"
            "- **Pilling:** usually comes from too many layers, silicone-heavy formulas, not enough wait time, or rubbing. Use less product and let layers set.\n"
            "- **Vitamin C turning orange/brown:** often means oxidation; discard if color, smell, or texture changed strongly.\n"
            "- **Expiration:** check PAO symbols, smell, color, texture, separation, and discard anything suspicious.\n"
            "- **Patch testing:** apply a small amount to one area for 24-48 hours before using all over.\n"
            "- **Non-comedogenic:** means designed to be less pore-clogging, but it is not a guarantee.\n"
            "- **Comedogenic ratings:** imperfect; the full formula matters more than one ingredient rating.\n"
            "- **Alcohol/natural ingredients:** context matters. Fatty alcohols can be moisturizing; natural ingredients and essential oils can still irritate.\n"
            "- **Diet, stress, sleep:** they can influence acne and inflammation, but skincare basics and medical care still matter."
        )
        wants_products = False
    elif wants_products:
        form = requested_product_step(message)
        if form == "sunscreen" or "sunscreen" in text or "spf" in text:
            response = (
                "For **sunscreen recommendations**, I would prioritize formulas that match your constraints:\n\n"
                "- Choose **broad-spectrum SPF 30+**.\n"
                "- For oily or acne-prone skin, look for lightweight, gel, fluid, or non-comedogenic textures.\n"
                "- For sensitive skin, prioritize fragrance-free options and consider mineral or hybrid formulas.\n"
                "- If you mentioned avoiding ingredients, skip products that clearly feature those ingredients in the name or claims."
            )
        elif form == "moisturizer" or "moisturizer" in text:
            response = (
                "For **moisturizer recommendations**, focus on barrier support and your texture preference:\n\n"
                "- Dry/sensitive skin: ceramides, glycerin, panthenol, squalane, and fragrance-free creams.\n"
                "- Oily/acne-prone skin: lightweight gel-cream or lotion, non-comedogenic finish.\n"
                "- If you are fragrance-allergic, avoid fragrance/parfum and essential oils."
            )
        elif form == "cleanser" or "cleanser" in text:
            response = (
                "For **cleanser recommendations**, choose based on irritation risk:\n\n"
                "- Acne-prone or oily skin: gentle gel cleanser; salicylic acid cleanser only if tolerated.\n"
                "- Dry/sensitive skin: creamy or hydrating cleanser with no fragrance.\n"
                "- Avoid harsh scrubs and stripping cleansers that leave the skin tight."
            )
        elif form == "treatment" or "serum" in text or "brightening products" in text:
            avoid_note = " I will avoid niacinamide, fragrance, and exfoliating acids in the wording here." if (avoid_niacinamide or avoid_fragrance or avoid_exfoliating_acids) else ""
            response = (
                f"For **treatment/serum recommendations**, match the active to the goal.{avoid_note}\n\n"
                "- Dark spots/brightening: tranexamic acid, alpha arbutin, azelaic acid, or vitamin C if tolerated.\n"
                "- Acne/clogged pores: adapalene, azelaic acid, or salicylic acid if tolerated.\n"
                "- Sensitive skin: keep the formula fragrance-free and introduce one active at a time.\n"
                "- Use sunscreen daily so treatment results are not undone by UV exposure."
            )
        else:
            response = (
                "For **product recommendations**, I would keep the set balanced: gentle cleanser, targeted treatment, moisturizer, and sunscreen.\n\n"
                "I’ll prioritize products that match the requested category, budget, and avoid list, then keep the routine simple enough to actually use."
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
        products = filter_products_by_avoid_terms(products, avoid_terms)

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
