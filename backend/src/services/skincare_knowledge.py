import re
from typing import Any, Dict, List, Optional


CONCERN_KEYWORDS = {
    "brightening": ["brightening", "brighten", "glow", "glowing"],
    "dullness": ["dull", "dullness", "tired skin", "lack of glow"],
    "dark_spots": ["dark spot", "dark spots", "post acne mark", "post-acne mark", "marks"],
    "hyperpigmentation": ["hyperpigmentation", "melasma", "discoloration", "uneven pigment"],
    "anti_aging": ["anti aging", "anti-aging", "wrinkle", "wrinkles", "fine line", "fine lines", "aging"],
    "acne": [
        "acne",
        "pimple",
        "pimples",
        "breakout",
        "breakouts",
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
    "dry_skin": ["dry skin", "dry", "flaky", "dehydrated"],
    "oily_skin": ["oily skin", "oily", "shine", "sebum"],
    "sensitive_skin": ["sensitive skin", "sensitive", "stinging", "burning", "irritation"],
    "redness": ["redness", "red skin", "red face", "rosacea", "flushing"],
    "barrier_repair": ["barrier", "barrier repair", "damaged barrier", "skin barrier"],
}

SKIN_TYPE_KEYWORDS = {
    "dry": ["dry skin", "my skin is dry", "i have dry"],
    "oily": ["oily skin", "my skin is oily", "i have oily"],
    "sensitive": ["sensitive skin", "my skin is sensitive", "i have sensitive"],
    "combination": ["combination skin", "combo skin"],
    "normal": ["normal skin"],
    "acne-prone": ["acne-prone", "acne prone"],
}

ROUTINE_TERMS = ["routine", "regimen", "steps", "morning", "night", "am", "pm"]
PRODUCT_TERMS = [
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
INGREDIENT_TERMS = ["ingredient", "ingredients", "active", "actives", "what should i use"]
SAFETY_TERMS = [
    "pain",
    "painful",
    "bleeding",
    "infection",
    "infected",
    "swollen",
    "pus",
    "spreading",
    "severe",
    "rash",
    "blister",
    "blistering",
    "scarring",
    "white patches",
    "red hot",
    "hot and swollen",
    "stop using",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def history_text(conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    if not conversation_history:
        return ""
    parts = []
    for message in conversation_history[-12:]:
        content = str(message.get("content", "")).split("Recommended Products")[0].strip()
        if content:
            parts.append(content)
    return normalize_text(" ".join(parts))


def infer_concerns(message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> List[str]:
    current = normalize_text(message)
    combined = normalize_text(f"{history_text(conversation_history)} {current}")
    concerns: List[str] = []
    for concern, keywords in CONCERN_KEYWORDS.items():
        if any(keyword in combined for keyword in keywords):
            concerns.append(concern)

    if "skin concern is dull skin" in current and "dullness" not in concerns:
        concerns.append("dullness")
    return concerns


def infer_skin_type(message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    combined = normalize_text(f"{history_text(conversation_history)} {message}")
    for skin_type, keywords in SKIN_TYPE_KEYWORDS.items():
        if any(keyword in combined for keyword in keywords):
            return skin_type
    return None


def infer_chat_intent(message: str) -> Dict[str, bool]:
    text = normalize_text(message)
    ingredient_question = any(term in text for term in INGREDIENT_TERMS)
    routine_request = any(term in text for term in ROUTINE_TERMS if term not in {"am", "pm"}) or bool(
        re.search(r"\b(?:am|pm)\b", text)
    )
    serum_request = any(term in text for term in ["serum", "serums", "ampoule"])
    sunscreen_request = any(term in text for term in ["sunscreen", "spf", "sun cream", "sun serum"])
    form_request = any(term in text for term in PRODUCT_TERMS)
    explicit_shopping = any(
        term in text
        for term in [
            "recommend",
            "recommendation",
            "product",
            "products",
            "buy",
            "affordable",
            "under $",
            "best cleanser",
            "best moisturizer",
            "best sunscreen",
            "what cleanser should i use",
            "what moisturizer should i use",
            "what sunscreen should i use",
        ]
    )
    informational_sunscreen = any(
        term in text
        for term in [
            "how much sunscreen",
            "how often",
            "do i need sunscreen",
            "can i skip sunscreen",
            "is spf",
            "does tinted sunscreen",
            "mineral vs chemical",
        ]
    )
    product_request = (explicit_shopping or (form_request and "should i use" in text)) and not informational_sunscreen
    safety_or_medical = any(term in text for term in SAFETY_TERMS)
    followup = len(text.split()) <= 5 and not routine_request

    if ingredient_question and not product_request:
        product_request = False

    return {
        "routine_request": routine_request,
        "ingredient_question": ingredient_question,
        "product_request": product_request,
        "serum_request": serum_request,
        "sunscreen_request": sunscreen_request,
        "followup": followup,
        "safety_or_medical": safety_or_medical,
    }


def get_retrieved_knowledge(message: str, concerns: List[str]) -> str:
    text = normalize_text(message)
    concern_set = set(concerns)
    snippets: List[str] = []

    if {"brightening", "dullness", "dark_spots", "hyperpigmentation"} & concern_set or "brightening" in text:
        snippets.append(
            "Brightening/dullness: Vitamin C is a morning antioxidant for glow and uneven tone. "
            "Niacinamide supports barrier health, dullness, redness, and oil balance and is beginner-friendly. "
            "Tranexamic acid targets dark spots, hyperpigmentation, and melasma-style uneven pigment. "
            "Alpha arbutin targets pigment and is usually gentle. Azelaic acid helps dark spots, acne-prone skin, and redness. "
            "AHAs like lactic/glycolic acid help surface dullness and texture 1-3 nights/week. "
            "Retinoids help texture, long-term tone, and post-acne marks at night with slow introduction. "
            "Sunscreen is required every morning for dark spots and brightening."
        )

    if "anti_aging" in concern_set:
        snippets.append(
            "Anti-aging: Use sunscreen daily, a retinoid at night, moisturizer/barrier support, optional peptides, "
            "and vitamin C in the morning. Start retinoids slowly."
        )

    if "acne" in concern_set:
        snippets.append(
            "Acne: Salicylic acid helps clogged pores. Benzoyl peroxide helps inflamed acne. "
            "Adapalene/retinoids help persistent acne and post-acne marks. Use non-comedogenic moisturizer and sunscreen."
        )

    if {"dry_skin", "sensitive_skin", "barrier_repair", "redness"} & concern_set:
        snippets.append(
            "Dry/sensitive/barrier: Use a gentle cleanser, ceramides, glycerin, hyaluronic acid, and petrolatum where appropriate. "
            "Pause acids and retinoids if skin is stinging or burning."
        )

    return "\n".join(snippets) or "Use general skincare principles: cleanse gently, treat the stated concern, moisturize, and use sunscreen daily."


def build_memory_updates(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    concerns = infer_concerns(message, conversation_history)
    skin_type = infer_skin_type(message, conversation_history)
    text = normalize_text(message)

    goals = []
    if "bright" in text or "glow" in text:
        goals.append("brightening")
    if "anti aging" in text or "anti-aging" in text:
        goals.append("anti-aging")

    preferences = []
    for brand in ["anua", "cosrx", "beauty of joseon", "medicube", "skin1004", "korean", "k-beauty"]:
        if brand in text:
            preferences.append(brand)

    avoid = []
    if "fragrance-free" in text or "fragrance free" in text or "avoid fragrance" in text:
        avoid.append("fragrance")
    avoid_match = re.search(r"avoid\s+([a-z0-9 ,+-]+)", text)
    if avoid_match:
        avoid.extend([item.strip() for item in re.split(r",| and | or ", avoid_match.group(1)) if item.strip()])
    without_match = re.search(r"(?:without|no)\s+([a-z0-9 ,+-]+)", text)
    if without_match:
        avoid.extend([item.strip() for item in re.split(r",| and | or ", without_match.group(1)) if item.strip()])
    intolerance_match = re.search(
        r"(?:cannot tolerate|cannot use|can't tolerate|can't use|cant tolerate|cant use|can not tolerate|can not use|allergic to|sensitive to|react to|reacts badly to|react badly to|breaks me out)\s+([a-z0-9 ,+-]+)",
        text,
    )
    if intolerance_match:
        avoid.extend(
            item.strip()
            for item in re.split(r",| and | or ", intolerance_match.group(1))
            if item.strip()
        )
    for ingredient in [
        "niacinamide",
        "vitamin c",
        "retinol",
        "retinoid",
        "glycolic acid",
        "salicylic acid",
        "benzoyl peroxide",
        "fragrance",
    ]:
        if ingredient in text and any(
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
            ]
        ):
            avoid.append(ingredient)

    return {
        "skin_type": skin_type,
        "concerns": concerns,
        "goals": list(dict.fromkeys(goals)),
        "preferences": list(dict.fromkeys(preferences)),
        "avoid": list(dict.fromkeys(avoid)),
    }
