import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.services.product_recommender import ProductRecommender  # noqa: E402


OUTPUT_DIR = ROOT / "backend" / "evaluation"
RESULTS_PATH = OUTPUT_DIR / "recommender_eval_results.json"
K = 8
RANDOM_STATE = 42

EVAL_SCENARIOS = [
    {
        "condition": "acne",
        "budget": 30,
        "keywords": ["acne", "pimple", "blemish", "salicylic", "benzoyl", "adapalene", "non-comedogenic"],
        "target_categories": ["cleanser", "treatment", "moisturizer"],
    },
    {
        "condition": "hyperpigmentation",
        "budget": 35,
        "keywords": ["dark", "spot", "hyperpigmentation", "brightening", "vitamin c", "niacinamide", "azelaic"],
        "target_categories": ["cleanser", "treatment", "moisturizer", "sunscreen"],
    },
    {
        "condition": "dry_skin",
        "budget": 30,
        "keywords": ["dry", "hydrating", "moisturizer", "ceramide", "hyaluronic", "glycerin"],
        "target_categories": ["cleanser", "treatment", "moisturizer"],
    },
    {
        "condition": "oily_skin",
        "budget": 30,
        "keywords": ["oily", "oil", "shine", "salicylic", "niacinamide", "gel", "lightweight"],
        "target_categories": ["cleanser", "treatment", "moisturizer"],
    },
    {
        "condition": "sensitive_skin",
        "budget": 35,
        "keywords": ["sensitive", "gentle", "barrier", "ceramide", "soothing", "fragrance-free"],
        "target_categories": ["cleanser", "treatment", "moisturizer", "sunscreen"],
    },
    {
        "condition": "melasma",
        "budget": 40,
        "keywords": ["melasma", "dark", "spot", "brightening", "tranexamic", "azelaic", "sunscreen"],
        "target_categories": ["cleanser", "treatment", "moisturizer", "sunscreen"],
    },
]

CATEGORY_KEYWORDS = {
    "cleanser": ["cleanser", "face wash", "wash", "cleansing", "micellar"],
    "toner": ["toner", "essence"],
    "treatment": [
        "serum",
        "ampoule",
        "treatment",
        "retinol",
        "retinal",
        "adapalene",
        "benzoyl",
        "salicylic",
        "azelaic",
        "tranexamic",
        "vitamin c",
    ],
    "moisturizer": ["moisturizer", "cream", "lotion", "gel cream", "barrier balm"],
    "sunscreen": ["sunscreen", "spf", "sun cream", "sun serum", "sun stick"],
    "mask": ["mask", "peel"],
    "patch": ["patch", "sticker"],
}


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def product_text(product: Dict[str, Any]) -> str:
    fields = [
        product.get("title"),
        product.get("name"),
        product.get("category"),
        product.get("directions"),
        product.get("condition_tag"),
    ]
    return " ".join(str(field or "") for field in fields).lower()


def infer_category(product: Dict[str, Any]) -> str:
    existing = str(product.get("category") or "").lower()
    if existing in CATEGORY_KEYWORDS:
        return existing

    text = product_text(product)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def flatten_catalog(recommender: ProductRecommender) -> pd.DataFrame:
    frames = []
    for condition, df in recommender.products.items():
        copy = df.copy()
        copy["condition_tag"] = condition
        frames.append(copy)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["asin", "title"])


def normalize_products(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    records = df.to_dict("records")
    normalized = []
    for record in records:
        record["price_numeric"] = safe_number(record.get("price_numeric"))
        record["rating"] = safe_number(record.get("rating"))
        record["reviews"] = safe_number(record.get("reviews"))
        record["inferred_category"] = infer_category(record)
        normalized.append(record)
    return normalized


def sort_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_columns = [column for column in ["rating", "reviews"] if column in df.columns]
    if not sort_columns:
        return df
    return df.sort_values(sort_columns, ascending=[False] * len(sort_columns))


def random_products(all_products: pd.DataFrame, _scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    sample_size = min(K, len(all_products))
    return normalize_products(all_products.sample(n=sample_size, random_state=RANDOM_STATE))


def top_rated_products(all_products: pd.DataFrame, _scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    return normalize_products(sort_by_quality(all_products).head(K))


def condition_aware_products(recommender: ProductRecommender, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    df = recommender.get_combined_products([scenario["condition"]])
    return normalize_products(sort_by_quality(df).head(K))


def budget_aware_products(recommender: ProductRecommender, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    df = recommender.get_combined_products([scenario["condition"]], budget_max=scenario["budget"])
    if df.empty:
        df = recommender.get_combined_products([scenario["condition"]])
    df = sort_by_quality(df)

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    records = normalize_products(df)

    for category in scenario["target_categories"]:
        matches = [product for product in records if product["inferred_category"] == category]
        for product in matches[:2]:
            product_id = product.get("asin") or product.get("title")
            if product_id not in selected_ids:
                selected.append(product)
                selected_ids.add(product_id)
            if len(selected) >= K:
                return selected

    for product in records:
        product_id = product.get("asin") or product.get("title")
        if product_id not in selected_ids:
            selected.append(product)
            selected_ids.add(product_id)
        if len(selected) >= K:
            return selected
    return selected


def top_k_relevance(products: Iterable[Dict[str, Any]], scenario: Dict[str, Any]) -> float:
    products = list(products)
    if not products:
        return 0.0
    keywords = [keyword.lower() for keyword in scenario["keywords"]]
    relevant_count = 0
    for product in products:
        text = product_text(product)
        condition_hit = scenario["condition"] in str(product.get("condition_tag", "")).lower()
        keyword_hit = any(keyword in text for keyword in keywords)
        if condition_hit or keyword_hit:
            relevant_count += 1
    return relevant_count / len(products)


def budget_fit_rate(products: Iterable[Dict[str, Any]], budget: float) -> float:
    products = list(products)
    priced = [product for product in products if safe_number(product.get("price_numeric")) > 0]
    if not priced:
        return 0.0
    return sum(safe_number(product.get("price_numeric")) <= budget for product in priced) / len(priced)


def category_coverage(products: Iterable[Dict[str, Any]], target_categories: List[str]) -> float:
    products = list(products)
    if not products or not target_categories:
        return 0.0
    covered = {infer_category(product) for product in products}
    return len(set(target_categories) & covered) / len(target_categories)


def average_rating(products: Iterable[Dict[str, Any]]) -> float:
    ratings = [safe_number(product.get("rating")) for product in products if safe_number(product.get("rating")) > 0]
    return sum(ratings) / len(ratings) if ratings else 0.0


def price_diversity(products: Iterable[Dict[str, Any]]) -> float:
    prices = [safe_number(product.get("price_numeric")) for product in products if safe_number(product.get("price_numeric")) > 0]
    if len(prices) < 2:
        return 0.0
    mean_price = sum(prices) / len(prices)
    if mean_price == 0:
        return 0.0
    variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
    return math.sqrt(variance) / mean_price


def score_strategy(products: List[Dict[str, Any]], scenario: Dict[str, Any]) -> Dict[str, float]:
    return {
        "top_k_relevance": round(top_k_relevance(products, scenario), 3),
        "budget_fit_rate": round(budget_fit_rate(products, scenario["budget"]), 3),
        "category_coverage": round(category_coverage(products, scenario["target_categories"]), 3),
        "average_rating": round(average_rating(products), 3),
        "price_diversity": round(price_diversity(products), 3),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    strategies = sorted({result["strategy"] for result in results})
    metrics = [
        "top_k_relevance",
        "budget_fit_rate",
        "category_coverage",
        "average_rating",
        "price_diversity",
    ]
    for strategy in strategies:
        rows = [result["metrics"] for result in results if result["strategy"] == strategy]
        summary[strategy] = {
            metric: round(sum(row[metric] for row in rows) / len(rows), 3)
            for metric in metrics
        }
    return summary


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    headers = ["strategy", "top-k relevance", "budget fit", "category coverage", "avg rating", "price diversity"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for strategy, metrics in summary.items():
        print(
            " | ".join(
                [
                    strategy,
                    f"{metrics['top_k_relevance']:.3f}",
                    f"{metrics['budget_fit_rate']:.3f}",
                    f"{metrics['category_coverage']:.3f}",
                    f"{metrics['average_rating']:.3f}",
                    f"{metrics['price_diversity']:.3f}",
                ]
            )
        )


def main() -> int:
    recommender = ProductRecommender()
    all_products = flatten_catalog(recommender)
    strategy_functions = {
        "random_products": lambda scenario: random_products(all_products, scenario),
        "top_rated_products": lambda scenario: top_rated_products(all_products, scenario),
        "condition_aware_recommender": lambda scenario: condition_aware_products(recommender, scenario),
        "budget_aware_recommender": lambda scenario: budget_aware_products(recommender, scenario),
    }

    results = []
    for scenario in EVAL_SCENARIOS:
        for strategy, selector in strategy_functions.items():
            products = selector(scenario)
            results.append(
                {
                    "condition": scenario["condition"],
                    "budget": scenario["budget"],
                    "strategy": strategy,
                    "product_count": len(products),
                    "metrics": score_strategy(products, scenario),
                    "sample_products": [
                        {
                            "title": product.get("title") or product.get("name"),
                            "price": product.get("price_numeric"),
                            "rating": product.get("rating"),
                            "category": product.get("inferred_category"),
                        }
                        for product in products[:3]
                    ],
                }
            )

    payload = {
        "k": K,
        "random_state": RANDOM_STATE,
        "metric_definitions": {
            "top_k_relevance": "Fraction of returned products matching the scenario condition tag or concern keywords.",
            "budget_fit_rate": "Fraction of returned products with price at or below the scenario budget.",
            "category_coverage": "Fraction of target routine categories represented by returned products.",
            "average_rating": "Mean catalog rating for returned products.",
            "price_diversity": "Coefficient of variation for returned product prices; higher means more varied price points.",
        },
        "summary": summarize(results),
        "scenario_results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload["summary"])
    print(f"\nSaved detailed results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
