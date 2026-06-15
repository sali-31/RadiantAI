import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["ENABLE_LIVE_PRODUCT_SEARCH"] = "false"

from backend.src.main import local_chat_response  # noqa: E402


EVAL_PATH = ROOT / "backend" / "tests" / "chatbot_eval_questions.json"


def normalize(value: str) -> str:
    return value.lower().replace("-", " ")


def run_case(case: dict) -> tuple[bool, list[str]]:
    response = local_chat_response(
        case["input"],
        conversation_history=case.get("history"),
    )
    text = response.get("response", "")
    normalized_text = normalize(text)
    failures = []

    for expected in case.get("expected", []):
        if normalize(expected) not in normalized_text:
            failures.append(f"missing expected text: {expected}")

    for blocked in case.get("should_not_include", []):
        if normalize(blocked) in normalized_text:
            failures.append(f"included blocked text: {blocked}")

    if case.get("products_expected") and not response.get("products"):
        failures.append("expected product recommendations")

    comparison_prompt = case.get("should_not_be_same_as")
    if comparison_prompt:
        comparison = local_chat_response(comparison_prompt).get("response", "")
        if normalize(comparison) == normalized_text:
            failures.append(f"same response as: {comparison_prompt}")

    return not failures, failures


def main() -> int:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    all_passed = True
    for index, case in enumerate(cases, start=1):
        passed, failures = run_case(case)
        status = "PASS" if passed else "FAIL"
        print(f"{status} {index}: {case['input']}")
        for failure in failures:
            print(f"  - {failure}")
        all_passed = all_passed and passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
