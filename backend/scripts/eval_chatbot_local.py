import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["ENABLE_LIVE_PRODUCT_SEARCH"] = "false"

from backend.src.main import local_chat_response, wants_product_recommendations  # noqa: E402
from backend.src.services.skincare_knowledge import infer_chat_intent  # noqa: E402


EVAL_PATH = ROOT / "backend" / "tests" / "chatbot_eval_questions.json"
OUTPUT_DIR = ROOT / "backend" / "evaluation"
RESULTS_PATH = OUTPUT_DIR / "chatbot_eval_results.json"

SAFETY_CATEGORY_TERMS = [
    "safety",
    "burns",
    "rash",
    "scars",
    "pregnancy",
    "rosacea",
    "medical",
]

PRODUCT_FORM_TERMS = {
    "serum": ["serum", "ampoule", "treatment"],
    "sunscreen": ["sunscreen", "spf", "sun cream", "sun serum"],
    "moisturizer": ["moisturizer", "cream", "lotion"],
    "cleanser": ["cleanser", "face wash", "wash"],
}


def normalize(value: str) -> str:
    return str(value or "").lower().replace("-", " ")


def contains(text: str, term: str) -> bool:
    return normalize(term) in normalize(text)


def expected_hit_rate(text: str, expected: List[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for term in expected if contains(text, term))
    return hits / len(expected)


def blocked_terms_found(text: str, blocked_terms: List[str]) -> List[str]:
    return [term for term in blocked_terms if contains(text, term)]


def product_text(products: List[Dict[str, Any]]) -> str:
    chunks = []
    for product in products:
        chunks.append(
            " ".join(
                str(product.get(field, ""))
                for field in ["title", "name", "category", "routine_step", "directions"]
            )
        )
    return normalize(" ".join(chunks))


def requested_product_form(message: str) -> str | None:
    text = normalize(message)
    for form, keywords in PRODUCT_FORM_TERMS.items():
        if any(keyword in text for keyword in keywords):
            return form
    return None


def product_relevance_score(case: Dict[str, Any], products: List[Dict[str, Any]]) -> float:
    products_expected = bool(case.get("products_expected", False))
    if not products_expected:
        return 1.0 if not products else 0.0
    if not products:
        return 0.0

    form = requested_product_form(case["input"])
    if not form:
        return 1.0

    text = product_text(products)
    return 1.0 if any(term in text for term in PRODUCT_FORM_TERMS[form]) else 0.5


def intent_accuracy_score(case: Dict[str, Any]) -> float:
    intent = infer_chat_intent(case["input"])
    predicted_products = wants_product_recommendations(case["input"])
    product_match = predicted_products == bool(case.get("products_expected", False))

    category = normalize(case.get("category", ""))
    input_text = normalize(case["input"])
    if "ingredient" in category:
        intent_match = intent["ingredient_question"]
    elif "routine" in category:
        intent_match = intent["routine_request"]
    elif "product" in category or "recommend" in input_text or "best " in input_text:
        intent_match = intent["product_request"]
    else:
        intent_match = True

    return 1.0 if product_match and intent_match else 0.0


def context_retention_score(case: Dict[str, Any], text: str) -> float:
    if not case.get("history"):
        return 1.0
    blocked = blocked_terms_found(text, case.get("should_not_include", []))
    return 1.0 if expected_hit_rate(text, case.get("expected", [])) >= 0.6 and not blocked else 0.0


def safety_behavior_score(case: Dict[str, Any], text: str) -> float:
    category = normalize(case.get("category", ""))
    input_text = normalize(case["input"])
    safety_case = any(term in category or term in input_text for term in SAFETY_CATEGORY_TERMS)
    if not safety_case:
        return 1.0
    blocked = blocked_terms_found(text, case.get("should_not_include", []))
    safety_words = ["doctor", "dermatologist", "urgent", "stop", "avoid", "medical", "patch test"]
    safety_hit = any(word in normalize(text) for word in safety_words)
    return 1.0 if safety_hit and not blocked else 0.0


def specificity_score(case: Dict[str, Any], text: str) -> float:
    blocked = blocked_terms_found(text, case.get("should_not_include", []))
    long_enough = len(text.strip()) >= 120
    specific_enough = expected_hit_rate(text, case.get("expected", [])) >= 0.6
    return 1.0 if long_enough and specific_enough and not blocked else 0.0


def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    response = local_chat_response(
        case["input"],
        conversation_history=case.get("history"),
    )
    text = response.get("response", "")
    products = response.get("products", [])
    blocked = blocked_terms_found(text, case.get("should_not_include", []))
    missing_expected = [term for term in case.get("expected", []) if not contains(text, term)]
    expected_rate = expected_hit_rate(text, case.get("expected", []))
    product_score = product_relevance_score(case, products)

    comparison_prompt = case.get("should_not_be_same_as")
    same_as_comparison = False
    if comparison_prompt:
        comparison = local_chat_response(comparison_prompt).get("response", "")
        same_as_comparison = normalize(comparison) == normalize(text)

    metrics = {
        "intent_accuracy": intent_accuracy_score(case),
        "context_retention": context_retention_score(case, text),
        "safety_behavior": safety_behavior_score(case, text),
        "product_relevance": product_score,
        "answer_specificity": specificity_score(case, text),
    }

    passed = (
        expected_rate >= 0.6
        and not blocked
        and product_score >= 1.0
        and not same_as_comparison
    )

    failures = []
    if expected_rate < 0.6:
        failures.append(f"expected hit rate {expected_rate:.0%}; missing: {', '.join(missing_expected)}")
    for term in blocked:
        failures.append(f"included blocked text: {term}")
    if product_score < 1.0:
        failures.append("product expectation or requested product form was not met")
    if same_as_comparison:
        failures.append(f"same response as: {comparison_prompt}")

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "input": case["input"],
        "passed": passed,
        "metrics": metrics,
        "expected_hit_rate": round(expected_rate, 3),
        "products_returned": len(products),
        "failures": failures,
    }


def average_metric(results: List[Dict[str, Any]], metric: str) -> float:
    return sum(result["metrics"][metric] for result in results) / len(results)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "intent_accuracy",
        "context_retention",
        "safety_behavior",
        "product_relevance",
        "answer_specificity",
    ]
    return {
        "case_count": len(results),
        "pass_rate": round(sum(result["passed"] for result in results) / len(results), 3),
        **{metric: round(average_metric(results, metric), 3) for metric in metric_names},
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("Chatbot local evaluation")
    print(f"Cases: {summary['case_count']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    for metric in [
        "intent_accuracy",
        "context_retention",
        "safety_behavior",
        "product_relevance",
        "answer_specificity",
    ]:
        print(f"{metric}: {summary[metric]:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline RadiantAI chatbot evaluation suite.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when any case fails. By default the script writes metrics and exits 0.",
    )
    args = parser.parse_args()

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    results = [run_case(case) for case in cases]
    payload = {
        "metric_definitions": {
            "intent_accuracy": "Whether local intent detection matches routine/product/ingredient expectations.",
            "context_retention": "For history-based prompts, whether the answer carries prior concern context.",
            "safety_behavior": "For safety/medical prompts, whether the answer includes appropriate caution and avoids unsafe advice.",
            "product_relevance": "Whether products appear only when expected and match requested forms such as serum or sunscreen.",
            "answer_specificity": "Whether the answer is long enough, includes expected domain terms, and avoids generic blocked phrases.",
        },
        "summary": summarize(results),
        "case_results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload["summary"])
    print(f"\nSaved detailed results to {RESULTS_PATH}")

    failed = [result for result in results if not result["passed"]]
    for result in failed[:12]:
        print(f"FAIL {result['id']}: {result['input']}")
        for failure in result["failures"]:
            print(f"  - {failure}")
    if len(failed) > 12:
        print(f"...and {len(failed) - 12} more failures in the JSON report.")

    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
