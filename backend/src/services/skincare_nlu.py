import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from .skincare_knowledge import build_memory_updates, history_text, normalize_text


logger = logging.getLogger(__name__)

DEFAULT_NLU: Dict[str, Any] = {
    "intent": "general_question",
    "concerns": ["general_skincare"],
    "product_form": None,
    "needs_products": False,
    "answer_type": "advice",
    "confidence": 0.45,
    "memory_updates": {
        "skin_type": None,
        "concerns": [],
        "goals": [],
        "preferences": [],
        "avoid": [],
    },
}

CONCERN_PATTERNS = {
    "closed_comedones": [
        "closed comedone",
        "closed comedones",
        "comedone",
        "comedones",
        "clogged pore",
        "clogged pores",
        "tiny bumps",
        "forehead bumps",
    ],
    "acne": ["acne", "pimple", "pimples", "breakout", "breakouts", "blemish", "blackhead", "whitehead"],
    "brightening": ["brightening", "brighten", "glow", "glowing"],
    "dullness": ["dull", "dullness", "tired skin"],
    "dark_spots": ["dark spot", "dark spots", "post acne mark", "post-acne mark", "brown marks"],
    "hyperpigmentation": ["hyperpigmentation", "melasma", "discoloration", "uneven pigment"],
    "anti_aging": ["anti aging", "anti-aging", "wrinkle", "wrinkles", "fine line", "fine lines", "aging"],
    "sensitive_skin": ["sensitive", "burn", "burns", "stinging", "irritation"],
    "redness": ["redness", "red skin", "red face", "rosacea", "flushing"],
    "dry_skin": ["dry", "flaky", "dehydrated"],
    "oily_skin": ["oily", "oil", "shine", "sebum"],
    "barrier_repair": ["barrier", "damaged barrier", "over exfoliation", "over-exfoliation"],
    "sunscreen": ["sunscreen", "spf", "sunblock"],
}

PRODUCT_FORM_PATTERNS = {
    "serum": ["serum", "serums", "ampoule"],
    "sunscreen": ["sunscreen", "spf", "sun cream", "sun serum", "sun stick", "sunstick"],
    "cleanser": ["cleanser", "face wash", "wash"],
    "toner": ["toner", "essence"],
    "moisturizer": ["moisturizer", "cream", "lotion"],
    "treatment": ["treatment", "retinol", "adapalene", "benzoyl", "salicylic"],
}

PRODUCT_REQUEST_TERMS = [
    "recommend",
    "recommendation",
    "product",
    "products",
    "buy",
    "best",
    "good",
    "which",
    "what should i buy",
]

SAFETY_TERMS = ["painful", "spreading", "swollen", "pus", "infection", "infected", "rash", "blister", "bleeding"]


class SkincareNLU:
    """Classify skincare chat messages before answer generation."""

    def __init__(self) -> None:
        self.model = None
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    os.getenv("GEMINI_NLU_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
                    generation_config={
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "max_output_tokens": 500,
                        "response_mime_type": "application/json",
                    },
                )
            except TypeError:
                self.model = genai.GenerativeModel(
                    os.getenv("GEMINI_NLU_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
                    generation_config={
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "max_output_tokens": 500,
                    },
                )
            except Exception as exc:
                logger.info("Gemini NLU unavailable; using heuristic classifier: %s", exc)

    def classify(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        skin_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        heuristic = self._heuristic_classify(message, conversation_history, skin_profile)
        if not self.model:
            return heuristic

        try:
            response = self.model.generate_content(self._build_prompt(message, conversation_history, skin_profile))
            parsed = self._parse_json(response.text)
            normalized = self._normalize(parsed, fallback=heuristic)
            if normalized["confidence"] < 0.55:
                return heuristic
            return normalized
        except Exception as exc:
            logger.info("Gemini NLU failed; using heuristic classifier: %s", exc)
            return heuristic

    def _build_prompt(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]],
        skin_profile: Optional[Dict[str, Any]],
    ) -> str:
        recent_history = history_text(conversation_history)
        return f"""Classify the current skincare chat message for RadiantAI.

Rules:
- The current user message has priority over conversation history.
- Use history only when the current message is a short follow-up with missing context.
- Do not classify an older topic if the current message asks something different.
- Return JSON only.

Allowed intents:
routine_request, ingredient_question, product_recommendation, application_timing, safety_or_medical, general_question, followup

Allowed answer_type:
routine, ingredients, products, timing, safety, advice

Known concerns include:
closed_comedones, acne, brightening, dullness, dark_spots, hyperpigmentation, melasma, anti_aging, sensitive_skin, redness, dry_skin, oily_skin, barrier_repair, sunscreen, general_skincare

Product forms include:
serum, sunscreen, cleanser, toner, moisturizer, treatment, routine_bundle, null

Examples:
- "closed comedones" => concerns ["closed_comedones", "acne"]
- "How long should I wait between skincare steps?" => intent "application_timing", answer_type "timing", needs_products false
- "ingredients for brightening" => intent "ingredient_question", needs_products false
- "recommend serums for brightening" => intent "product_recommendation", product_form "serum", needs_products true

Current message: {message}
Recent history, use only for short follow-ups: {recent_history or "none"}
Skin profile: {json.dumps(skin_profile or {}, ensure_ascii=True)}

Schema:
{{
  "intent": "routine_request",
  "concerns": ["closed_comedones", "acne"],
  "product_form": "routine_bundle",
  "needs_products": true,
  "answer_type": "routine",
  "confidence": 0.9,
  "memory_updates": {{
    "skin_type": null,
    "concerns": [],
    "goals": [],
    "preferences": [],
    "avoid": []
  }}
}}
"""

    def _heuristic_classify(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        skin_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = normalize_text(message)
        current_concerns = self._extract_concerns(text)
        is_short_followup = len(text.split()) <= 5 and not self._has_explicit_intent(text)

        if current_concerns:
            concerns = current_concerns
        elif is_short_followup:
            concerns = self._extract_concerns(history_text(conversation_history))
            if not concerns and skin_profile:
                concerns = list(skin_profile.get("concerns") or [])
        else:
            concerns = []

        if not concerns:
            concerns = ["general_skincare"]

        product_form = self._extract_product_form(text)
        intent = self._infer_intent(text, product_form)
        answer_type = {
            "routine_request": "routine",
            "ingredient_question": "ingredients",
            "product_recommendation": "products",
            "application_timing": "timing",
            "safety_or_medical": "safety",
        }.get(intent, "advice")
        needs_products = intent == "product_recommendation" or intent == "routine_request"

        if intent == "routine_request":
            product_form = product_form or "routine_bundle"

        memory_updates = build_memory_updates(message, conversation_history)
        memory_updates["concerns"] = [concern for concern in concerns if concern != "general_skincare"]

        return self._normalize(
            {
                "intent": intent,
                "concerns": concerns,
                "product_form": product_form,
                "needs_products": needs_products,
                "answer_type": answer_type,
                "confidence": 0.82 if current_concerns or intent != "general_question" else 0.62,
                "memory_updates": memory_updates,
            },
            fallback=DEFAULT_NLU,
        )

    def _extract_concerns(self, text: str) -> List[str]:
        concerns: List[str] = []
        for concern, patterns in CONCERN_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                concerns.append(concern)
        if "closed_comedones" in concerns and "acne" not in concerns:
            concerns.append("acne")
        if "melasma" in text and "hyperpigmentation" not in concerns:
            concerns.append("hyperpigmentation")
        return list(dict.fromkeys(concerns))

    def _extract_product_form(self, text: str) -> Optional[str]:
        for form, patterns in PRODUCT_FORM_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                return form
        return None

    def _has_explicit_intent(self, text: str) -> bool:
        intent_terms = [
            "routine",
            "regimen",
            "ingredient",
            "ingredients",
            "recommend",
            "recommendation",
            "product",
            "products",
            "how long should i wait",
            "wait between",
            "between skincare steps",
        ]
        return any(term in text for term in intent_terms) or bool(self._extract_product_form(text))

    def _infer_intent(self, text: str, product_form: Optional[str]) -> str:
        if any(term in text for term in SAFETY_TERMS):
            return "safety_or_medical"
        if any(term in text for term in ["wait between", "how long should i wait", "between skincare steps", "between steps", "layer skincare", "skincare order timing"]):
            return "application_timing"
        if any(term in text for term in ["ingredient", "ingredients", "active", "actives", "what should i use"]):
            explicit_product = any(term in text for term in PRODUCT_REQUEST_TERMS) or bool(product_form)
            return "product_recommendation" if explicit_product and any(term in text for term in ["recommend", "product", "buy"]) else "ingredient_question"
        if any(term in text for term in ["routine", "regimen", "morning", "night", "am", "pm"]):
            return "routine_request"
        if any(term in text for term in PRODUCT_REQUEST_TERMS) or product_form:
            return "product_recommendation"
        if len(text.split()) <= 5:
            return "followup"
        return "general_question"

    def _parse_json(self, text: str) -> Dict[str, Any]:
        clean = str(text or "").strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?", "", clean, flags=re.IGNORECASE).strip()
            clean = re.sub(r"```$", "", clean).strip()
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize(self, data: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        normalized = json.loads(json.dumps(fallback))
        normalized.update({key: value for key, value in data.items() if value is not None})

        concerns = normalized.get("concerns") or ["general_skincare"]
        if isinstance(concerns, str):
            concerns = [concerns]
        normalized["concerns"] = list(dict.fromkeys(str(concern) for concern in concerns if concern)) or ["general_skincare"]
        if "closed_comedones" in normalized["concerns"] and "acne" not in normalized["concerns"]:
            normalized["concerns"].append("acne")

        normalized["intent"] = str(normalized.get("intent") or "general_question")
        normalized["answer_type"] = str(normalized.get("answer_type") or "advice")
        normalized["product_form"] = normalized.get("product_form") or None
        normalized["needs_products"] = bool(normalized.get("needs_products", False))
        try:
            normalized["confidence"] = float(normalized.get("confidence", 0.5))
        except (TypeError, ValueError):
            normalized["confidence"] = 0.5

        memory_updates = normalized.get("memory_updates") or {}
        if not isinstance(memory_updates, dict):
            memory_updates = {}
        normalized["memory_updates"] = {
            "skin_type": memory_updates.get("skin_type"),
            "concerns": list(memory_updates.get("concerns") or []),
            "goals": list(memory_updates.get("goals") or []),
            "preferences": list(memory_updates.get("preferences") or []),
            "avoid": list(memory_updates.get("avoid") or []),
        }
        return normalized
