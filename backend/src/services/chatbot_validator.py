import json
import re
from typing import Any, Dict, List, Optional


AVOID_PATTERNS = [
    "cannot tolerate",
    "cannot use",
    "can't tolerate",
    "can't use",
    "cant tolerate",
    "cant use",
    "can not tolerate",
    "can not use",
    "allergic to",
    "sensitive to",
    "react to",
    "reacts badly to",
    "react badly to",
    "breaks me out",
    "without",
    "avoid",
]

KNOWN_AVOID_TERMS = [
    "niacinamide",
    "vitamin c",
    "retinol",
    "retinoid",
    "retinoids",
    "salicylic acid",
    "benzoyl peroxide",
    "hyaluronic acid",
    "fragrance",
    "essential oils",
    "exfoliating acids",
    "glycolic acid",
    "lactic acid",
]

SAFETY_TERMS = [
    "swollen",
    "spreading",
    "painful",
    "pus",
    "red, hot",
    "hot and swollen",
    "blistering",
    "chemical burn",
    "bleeding",
    "deep scars",
]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def extract_response_text(value: Any) -> str:
    """Return user-facing text even if a model accidentally nested JSON."""
    if isinstance(value, dict):
        return extract_response_text(value.get("response_text") or value.get("response") or "")

    text = str(value or "").strip()
    for _ in range(3):
        if not text.startswith("{"):
            return text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'"response_text"\s*:\s*"(?P<text>.*?)(?<!\\)"', text, flags=re.DOTALL)
            if not match:
                return text
            text = match.group("text").replace('\\"', '"').replace("\\n", "\n").strip()
            continue
        if not isinstance(parsed, dict):
            return text
        next_text = parsed.get("response_text") or parsed.get("response")
        if not next_text:
            return text
        text = str(next_text).strip()
    return text


def extract_constraints(
    message: str,
    skin_profile: Optional[Dict[str, Any]] = None,
    nlu: Optional[Dict[str, Any]] = None,
    memory_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    text = normalize_text(message)
    avoid: List[str] = []

    for source in [skin_profile or {}, memory_updates or {}, (nlu or {}).get("memory_updates") or {}]:
        for key in ["avoid", "avoided_ingredients", "allergies", "sensitivities"]:
            for value in source.get(key) or []:
                clean = normalize_text(value).strip(" .")
                if clean and clean not in avoid:
                    avoid.append(clean)

    for term in KNOWN_AVOID_TERMS:
        if term in text and any(pattern in text for pattern in AVOID_PATTERNS):
            if term not in avoid:
                avoid.append(term)

    if "no fragrance" in text or "fragrance-free" in text or "fragrance free" in text:
        if "fragrance" not in avoid:
            avoid.append("fragrance")
    if "no exfoliating acid" in text or "without exfoliating acid" in text:
        if "exfoliating acids" not in avoid:
            avoid.append("exfoliating acids")

    return {"avoid": avoid}


def sentence_allows_avoid_mention(sentence: str, term: str) -> bool:
    sentence_text = normalize_text(sentence)
    if term not in sentence_text:
        return True
    allowed_context = [
        "avoid",
        "skip",
        "without",
        "do not use",
        "don't use",
        "cannot use",
        "cannot tolerate",
        "if tolerated",
        "if you tolerate",
        "if your skin tolerates",
    ]
    return any(context in sentence_text for context in allowed_context)


def remove_forbidden_recommendations(response_text: str, avoid_terms: List[str]) -> str:
    if not avoid_terms:
        return response_text

    sentences = re.split(r"(?<=[.!?])\s+|\n", response_text)
    kept: List[str] = []
    removed_terms: List[str] = []
    for sentence in sentences:
        sentence_text = normalize_text(sentence)
        bad_terms = [
            term
            for term in avoid_terms
            if term in sentence_text and not sentence_allows_avoid_mention(sentence, term)
        ]
        if bad_terms:
            removed_terms.extend(bad_terms)
            continue
        kept.append(sentence)

    repaired = "\n".join(part for part in kept if part.strip()).strip()
    if removed_terms:
        unique_terms = ", ".join(dict.fromkeys(removed_terms))
        repaired = (
            f"{repaired.rstrip()}\n\n"
            f"Note: I removed suggestions that conflicted with your avoid list: {unique_terms}."
        ).strip()
    return repaired or response_text


def filter_products(products: List[Dict[str, Any]], avoid_terms: List[str]) -> List[Dict[str, Any]]:
    if not products or not avoid_terms:
        return products

    filtered = []
    for product in products:
        product_text = normalize_text(
            " ".join(
                str(product.get(field, ""))
                for field in ["title", "name", "brand", "category", "routine_step", "directions", "reason"]
            )
        )
        if any(term in product_text for term in avoid_terms):
            continue
        filtered.append(product)
    return filtered


def should_have_products(message: str, nlu: Optional[Dict[str, Any]]) -> bool:
    text = normalize_text(message)
    if any(term in text for term in ["stop recommending products", "no longer want product", "just explain"]):
        return False
    if nlu and nlu.get("needs_products"):
        return True
    shopping_terms = ["recommend", "recommendation", "products", "buy", "affordable", "under $"]
    return any(term in text for term in shopping_terms)


def requires_safety_language(message: str, nlu: Optional[Dict[str, Any]]) -> bool:
    text = normalize_text(message)
    return bool(nlu and nlu.get("intent") == "safety_or_medical") or any(term in text for term in SAFETY_TERMS)


def validate_chat_response(
    *,
    message: str,
    response_text: Any,
    products: Optional[List[Dict[str, Any]]] = None,
    skin_profile: Optional[Dict[str, Any]] = None,
    nlu: Optional[Dict[str, Any]] = None,
    memory_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and repair the final chatbot payload before it reaches the UI."""
    issues: List[str] = []
    text = extract_response_text(response_text)
    if text != str(response_text or "").strip():
        issues.append("unwrapped_raw_json")

    constraints = extract_constraints(message, skin_profile=skin_profile, nlu=nlu, memory_updates=memory_updates)
    avoid_terms = constraints["avoid"]
    repaired_text = remove_forbidden_recommendations(text, avoid_terms)
    if repaired_text != text:
        issues.append("removed_forbidden_ingredient_recommendation")
        text = repaired_text

    final_products = list(products or [])
    filtered_products = filter_products(final_products, avoid_terms)
    if len(filtered_products) != len(final_products):
        issues.append("removed_products_matching_avoid_terms")
        final_products = filtered_products

    if not should_have_products(message, nlu) and final_products:
        issues.append("removed_unrequested_products")
        final_products = []

    if requires_safety_language(message, nlu):
        safety_text = normalize_text(text)
        if not any(term in safety_text for term in ["doctor", "dermatologist", "urgent", "medical", "stop"]):
            issues.append("added_safety_language")
            text = (
                f"{text.rstrip()}\n\n"
                "Because you described a possible red flag, stop irritating products and contact a doctor or dermatologist, especially if symptoms are painful, spreading, swollen, pus-filled, blistering, or worsening."
            )

    if normalize_text(message) and "response_text" in normalize_text(text)[:80]:
        issues.append("stripped_response_text_label")
        text = re.sub(r'^\s*"?response_text"?\s*:\s*', "", text, flags=re.IGNORECASE).strip()

    return {
        "response_text": text,
        "products": final_products,
        "constraints": constraints,
        "issues": issues,
    }
