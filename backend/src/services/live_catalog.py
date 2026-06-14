import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote_plus, urlencode, urljoin

import requests


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
VERIFIED_CATALOG_PATH = DATA_DIR / "verified_beauty_catalog.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

STEP_KEYWORDS = {
    "cleanser": ["cleanser", "cleansing", "face wash", "wash", "cleansing oil", "cleansing foam"],
    "toner": ["toner", "essence", "pad", "pads"],
    "treatment": [
        "serum",
        "ampoule",
        "treatment",
        "retinol",
        "retinal",
        "azelaic",
        "kojic",
        "tranexamic",
        "txa",
        "arbutin",
        "niacinamide",
        "vitamin c",
        "dark spot",
        "bha",
        "aha",
        "booster",
    ],
    "moisturizer": ["moisturizer", "cream", "lotion", "balm", "mask"],
    "sunscreen": ["sunscreen", "sun", "spf", "sunstick"],
}

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

BRAND_PRIORITY = [
    "Anua",
    "COSRX",
    "SKIN1004",
    "Beauty of Joseon",
    "Medicube",
    "Aestura",
    "Round Lab",
    "Dr.Althea",
    "Numbuzin",
    "Pyunkang Yul",
    "Laneige",
    "Drunk Elephant",
    "Tatcha",
    "Glow Recipe",
    "The Ordinary",
    "The Inkey List",
    "Paula's Choice",
]


def infer_step(product_name: str, fallback: str = "skincare") -> str:
    text = product_name.lower()
    if any(keyword in text for keyword in ["sunscreen", "spf", "sun serum", "sun cream", "sunstick"]):
        return "sunscreen"
    for step, keywords in STEP_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return step
    return fallback.lower().replace("skin care", "skincare") or "skincare"


def load_verified_catalog() -> List[Dict[str, str]]:
    if not VERIFIED_CATALOG_PATH.exists():
        return []
    with VERIFIED_CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def catalog_row_to_product(row: Dict[str, str]) -> Dict[str, Any]:
    product_name = row.get("product_name", "")
    brand = row.get("brand", "")
    retailer = row.get("retailer", "")
    category = infer_step(product_name, row.get("category", "skincare"))

    product_url = row.get("product_url") or search_url_for(retailer, brand, product_name)
    return {
        "title": f"{brand} {product_name}".strip(),
        "name": product_name,
        "brand": brand,
        "category": category,
        "routine_step": category if category in STEP_KEYWORDS else "",
        "price": row.get("sale_price") or row.get("price") or "See retailer",
        "rating": parse_float(row.get("rating")),
        "reviews": parse_review_count(row.get("review_count")),
        "link": product_url,
        "product_url": product_url,
        "thumbnail": row.get("image_url") or "",
        "image_url": row.get("image_url") or "",
        "retailer": retailer,
        "source": row.get("source_scope") or retailer,
        "barcode": row.get("barcode") or "",
        "product_id": row.get("product_id") or "",
        "sku_id": row.get("sku_id") or "",
        "reason": f"Verified {retailer} catalog product from {brand}.",
    }


def search_url_for(retailer: str, brand: str, product_name: str) -> str:
    query = quote_plus(f"{brand} {product_name}".strip())
    if "sephora" in retailer.lower():
        return f"https://www.sephora.com/search?keyword={query}"
    if "style" in retailer.lower() or "korean" in retailer.lower():
        return f"https://www.stylekorean.com/search/{query}"
    return ""


def search_verified_catalog(query: str, *, step: Optional[str] = None, limit: int = 12) -> List[Dict[str, Any]]:
    rows = load_verified_catalog()
    scored = []
    for row in rows:
        product = catalog_row_to_product(row)
        if step and product.get("category") != step and product.get("routine_step") != step:
            continue
        score = score_product(query, product)
        if score > 0 or not query:
            scored.append((score, product))

    scored.sort(key=lambda item: item[0], reverse=True)
    return dedupe_products([product for _, product in scored], limit=limit)


def products_for_step(step: str, *, query: str = "", limit: int = 2) -> List[Dict[str, Any]]:
    products = search_verified_catalog(query, step=step, limit=limit * 3)
    return products[:limit]


def search_live_catalog(query: str, *, limit: int = 12, timeout: float = 4.0) -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []
    products.extend(search_sephora_live(query, limit=limit, timeout=timeout))
    products.extend(search_stylekorean_live(query, limit=limit, timeout=timeout))
    return dedupe_products(products, limit=limit)


def search_sephora_live(query: str, *, limit: int, timeout: float) -> List[Dict[str, Any]]:
    endpoint = "https://www.sephora.com/api/catalog/search"
    params = {"keyword": query, "currentPage": 1, "pageSize": min(limit, 24)}
    try:
        response = requests.get(endpoint, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    raw_products = extract_product_lists(payload)
    products = []
    for raw in raw_products[:limit]:
        if not isinstance(raw, dict):
            continue
        sku = raw.get("currentSku") or raw.get("sku") or {}
        if not isinstance(sku, dict):
            sku = {}
        title = first_present(raw, ["displayName", "productName", "name"])
        brand = first_present(raw, ["brandName", "brand", "brandDisplayName"])
        product_url = urljoin("https://www.sephora.com", raw.get("targetUrl") or raw.get("productUrl") or "")
        image_url = raw.get("heroImage") or raw.get("image") or ""
        if not image_url and isinstance(sku.get("skuImages"), dict):
            image_url = sku["skuImages"].get("image450") or sku["skuImages"].get("image250") or ""
        products.append(
            {
                "title": f"{brand} {title}".strip(),
                "name": title,
                "brand": brand,
                "category": infer_step(title),
                "routine_step": infer_step(title),
                "price": first_present(sku, ["salePrice", "listPrice", "valuePrice"], "See retailer"),
                "rating": parse_float(first_present(raw, ["rating", "avgRating"])),
                "reviews": parse_review_count(first_present(raw, ["reviews", "reviewCount", "numberOfReviews"])),
                "link": product_url,
                "product_url": product_url,
                "thumbnail": image_url,
                "image_url": image_url,
                "retailer": "Sephora",
                "source": "Live Sephora search",
                "product_id": first_present(raw, ["productId", "id"]),
                "sku_id": first_present(sku, ["skuId", "id"]),
                "reason": "Live result from Sephora public search.",
            }
        )
    return products


def search_stylekorean_live(query: str, *, limit: int, timeout: float) -> List[Dict[str, Any]]:
    search_url = f"https://www.stylekorean.com/search/{quote_plus(query)}"
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        html = response.text
    except Exception:
        return []

    products = []
    seen_titles = set()
    product_patterns = [
        r"\[(?P<brand>[A-Za-z0-9 .&'+-]{2,40})\]\s*(?P<name>[^<\n{};=]{8,140})",
    ]
    for pattern in product_patterns:
        for match in re.finditer(pattern, html):
            brand = match.groupdict().get("brand") or ""
            name = clean_html_text(match.group("name"))
            if not is_plausible_stylekorean_product(brand, name, query):
                continue
            if not name or name.lower() in seen_titles:
                continue
            seen_titles.add(name.lower())
            products.append(
                {
                    "title": f"{brand} {name}".strip(),
                    "name": name,
                    "brand": brand,
                    "category": infer_step(name),
                    "routine_step": infer_step(name),
                    "price": "See retailer",
                    "link": search_url,
                    "product_url": search_url,
                    "thumbnail": "",
                    "image_url": "",
                    "retailer": "StyleKorean",
                    "source": "Live StyleKorean search",
                    "reason": "Live result from StyleKorean search.",
                }
            )
            if len(products) >= limit:
                return products
    return products


def is_plausible_stylekorean_product(brand: str, name: str, query: str) -> bool:
    blocked = ["http", "function", "facebook", "instagram", "tiktok", "google", "script"]
    text = f"{brand} {name}".lower()
    if any(token in text for token in blocked):
        return False
    if any(token in name for token in ["<", ">", "{", "}", "=", " style", '",', "','"]):
        return False
    if name.count(",") > 1:
        return False
    if len(name) < 8 or len(name) > 140:
        return False
    query_tokens = [token for token in re.split(r"[^a-z0-9]+", query.lower()) if len(token) > 2]
    if query_tokens and not any(token in text for token in query_tokens):
        return False
    return any(char.isalpha() for char in brand) and any(char.isalpha() for char in name)


def extract_product_lists(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    products = []
    for key in ["products", "productTiles", "items"]:
        value = payload.get(key)
        if isinstance(value, list):
            products.extend(value)
    for value in payload.values():
        if isinstance(value, dict):
            products.extend(extract_product_lists(value))
    return products


def score_product(query: str, product: Dict[str, Any]) -> int:
    title_haystack = " ".join(
        str(product.get(key, ""))
        for key in ["title", "name", "brand"]
    ).lower()
    haystack = " ".join(
        str(product.get(key, ""))
        for key in ["title", "name", "brand", "category", "retailer", "source"]
    ).lower()
    tokens = query_tokens(query)
    score = 0
    for token in tokens:
        token_weight = 10 if token in {"kojic", "tranexamic", "txa", "arbutin", "azelaic"} else 4
        if token in title_haystack:
            score += token_weight
        elif token in haystack:
            score += max(2, token_weight // 2)
    brand = product.get("brand", "")
    if brand in BRAND_PRIORITY:
        score += len(BRAND_PRIORITY) - BRAND_PRIORITY.index(brand)
    if product.get("retailer"):
        score += 2
    return score


def query_tokens(query: str) -> List[str]:
    tokens: List[str] = []
    for token in re.split(r"[^a-z0-9]+", query.lower()):
        if len(token) <= 2 or token in QUERY_STOPWORDS:
            continue
        tokens.append(token)
        if token.endswith("s") and len(token) > 4:
            tokens.append(token[:-1])
    return list(dict.fromkeys(tokens))


def dedupe_products(products: Iterable[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for product in products:
        key = (
            str(product.get("retailer", "")).lower(),
            str(product.get("brand", "")).lower(),
            str(product.get("name") or product.get("title", "")).lower(),
        )
        if key in seen or not key[2]:
            continue
        seen.add(key)
        out.append(product)
        if len(out) >= limit:
            break
    return out


def first_present(data: dict, keys: List[str], default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def parse_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def parse_review_count(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    text = str(value).strip().lower().replace(",", "")
    try:
        if text.endswith("k"):
            return int(float(text[:-1]) * 1000)
        return int(float(text))
    except Exception:
        return None


def clean_html_text(value: str) -> str:
    value = re.sub(r"\\u[0-9a-fA-F]{4}", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = json.loads(f'"{value}"') if "\\/" in value or '\\"' in value else value
    return re.sub(r"\s+", " ", value).strip()
