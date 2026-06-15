import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MEMORY_PATH = DATA_DIR / "chat_memory.json"


DEFAULT_MEMORY = {
    "skin_type": "",
    "concerns": [],
    "budget_max": None,
    "allergies": [],
    "sensitivities": [],
    "preferred_brands": [],
    "avoided_ingredients": [],
    "last_products": [],
}


def load_memory_store() -> Dict[str, Dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return {}
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory_store(store: Dict[str, Dict[str, Any]]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")


def get_user_memory(user_id: str) -> Dict[str, Any]:
    store = load_memory_store()
    memory = DEFAULT_MEMORY.copy()
    memory.update(store.get(user_id, {}))
    return memory


def update_user_memory(
    user_id: str,
    message: str,
    products: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    store = load_memory_store()
    memory = DEFAULT_MEMORY.copy()
    memory.update(store.get(user_id, {}))

    text = message.lower()
    for skin_type in ["oily", "dry", "sensitive", "combination", "normal", "acne-prone"]:
        if skin_type in text:
            memory["skin_type"] = skin_type

    concerns = {
        "acne": ["acne", "pimple", "breakout", "blemish"],
        "hyperpigmentation": ["hyperpigmentation", "dark spot", "dark spots", "melasma", "discoloration"],
        "redness": ["redness", "rosacea", "flushing"],
        "dryness": ["dry", "flaky", "dehydrated"],
        "oiliness": ["oily", "shine", "sebum"],
        "texture": ["texture", "bumps", "pores"],
    }
    for concern, keywords in concerns.items():
        if any(keyword in text for keyword in keywords):
            add_unique(memory, "concerns", concern)

    budget_match = re.search(r"(?:under|below|less than|max|budget)\s*\$?\s*(\d{1,4})", text)
    if budget_match:
        memory["budget_max"] = int(budget_match.group(1))

    allergy_match = re.search(r"(?:allergic to|allergy to|avoid)\s+([a-z0-9 ,+-]+)", text)
    if allergy_match:
        for value in split_list_text(allergy_match.group(1)):
            add_unique(memory, "allergies" if "allerg" in text else "avoided_ingredients", value)

    if "fragrance-free" in text or "fragrance free" in text:
        add_unique(memory, "avoided_ingredients", "fragrance")
        add_unique(memory, "sensitivities", "fragrance")

    brand_aliases = [
        "anua",
        "cosrx",
        "beauty of joseon",
        "medicube",
        "skin1004",
        "laneige",
        "dr. jart",
        "the ordinary",
        "paula's choice",
        "tatcha",
        "glow recipe",
    ]
    for brand in brand_aliases:
        if brand in text and any(term in text for term in ["like", "love", "prefer", "korean", "k-beauty"]):
            add_unique(memory, "preferred_brands", brand)

    if products:
        memory["last_products"] = [
            {
                "brand": product.get("brand", ""),
                "name": product.get("name") or product.get("title", ""),
                "category": product.get("routine_step") or product.get("category", ""),
                "retailer": product.get("retailer") or product.get("source") or product.get("data_source", ""),
            }
            for product in products[:8]
        ]

    store[user_id] = memory
    save_memory_store(store)
    return memory


def memory_summary(memory: Dict[str, Any]) -> str:
    parts = []
    if memory.get("skin_type"):
        parts.append(f"skin type: {memory['skin_type']}")
    if memory.get("concerns"):
        parts.append(f"concerns: {', '.join(memory['concerns'])}")
    if memory.get("budget_max"):
        parts.append(f"budget max: ${memory['budget_max']}")
    if memory.get("allergies"):
        parts.append(f"allergies: {', '.join(memory['allergies'])}")
    if memory.get("avoided_ingredients"):
        parts.append(f"avoid: {', '.join(memory['avoided_ingredients'])}")
    if memory.get("preferred_brands"):
        parts.append(f"preferred brands: {', '.join(memory['preferred_brands'])}")
    return "; ".join(parts) or "No saved user preferences yet."


def add_unique(memory: Dict[str, Any], key: str, value: str) -> None:
    value = value.strip(" .")
    if not value:
        return
    values = memory.setdefault(key, [])
    if value not in values:
        values.append(value)


def split_list_text(value: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r",| and | or ", value)
        if item.strip()
    ][:5]
