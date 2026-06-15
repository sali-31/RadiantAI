"""
Chatbot service for answering skincare and dermatology questions
Uses Google's Gemini API for intelligent responses
"""

import os
import logging
from typing import Optional, List, Dict
import google.generativeai as genai
import re
import json

logger = logging.getLogger(__name__)


class SkinHealthChatbot:
    """AI-powered chatbot for skin health questions"""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.35,
            "top_p": 0.9,
            "max_output_tokens": 1200,
            "response_mime_type": "application/json",
        }
        try:
            self.model = genai.GenerativeModel(
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                generation_config=generation_config,
            )
        except TypeError:
            generation_config.pop("response_mime_type", None)
            self.model = genai.GenerativeModel(
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                generation_config=generation_config,
            )

        self.system_prompt = """You are RadiantAI, a strong skincare AI assistant inside a consumer skincare app.

Answer the exact user question directly. The current user message has priority over conversation history. Use history only for short follow-ups. Do not answer an older topic if the current message asks something different. Never ask for the user's concern again if the concern is already present in the current message, skin profile, NLU classification, or conversation history.

You must distinguish between:
- routine requests: give AM/PM steps only when the user asks for a routine.
- ingredient questions: explain useful ingredients and how/when to use them. Do not force product cards.
- product recommendation requests: explain product-type logic and return clear searchable product names.
- follow-up messages: infer context from history, especially short messages like "dark spots", "dull skin", "what ingredients should I use", or "serums?".

Be specific for concerns: dark spots, dull skin, acne, anti-aging, dry skin, oily skin, sensitive skin, redness, hyperpigmentation, and brightening.

Use the provided NLU classification and retrieved skincare knowledge as the main grounding context. If NLU says the intent is application_timing, answer timing guidance, not a routine. If NLU says closed_comedones, answer clogged pores/comedonal acne, not anti-aging.

Product rules:
- If retrieved product candidates are provided, prefer those exact product names.
- Never invent retailer links, prices, or availability.
- Respect the user's avoid list from skin_profile, memory, and NLU memory_updates. If the user says they cannot tolerate an ingredient, do not recommend that ingredient except to say to avoid or skip it.
- For serum requests, prefer serum/ampoule/treatment products.
- For sunscreen requests, prefer SPF/sunscreen products.
- For routine requests, include cleanser, treatment/serum, moisturizer, sunscreen, and toner when useful.
- For routine + products requests, write a useful AM/PM routine in response_text and put product names in recommended_products.

Safety:
- Mention red flags briefly: painful, spreading, infected, severe, scarring, or persistent symptoms should be checked by a dermatologist.
- Do not overdo medical disclaimers.

Style:
- Be context-aware, specific, non-repetitive, and concise.
- Avoid generic "tell me your concern" responses when a concern exists.
- Do not reveal internal reasoning, drafting notes, uncertainty narration, or self-corrections.

Return valid JSON only with this schema:
{
  "response_text": "Markdown answer for the user. This value must be plain Markdown text only, never a JSON string and never an object.",
  "recommended_products": ["clear searchable product name 1", "clear searchable product name 2"],
  "product_query": "short optimized search query for the product engine",
  "wants_products": true,
  "needs_followup": false,
  "followup_questions": [],
  "memory_updates": {
    "skin_type": null,
    "concerns": [],
    "goals": [],
    "preferences": [],
    "avoid": []
  }
}
"""
        
    def _sanitize_input(self, text: str) -> tuple[bool, str]:
        """
        Remove PII and block malicious code patterns:
        -   This prevents accidental leakage of contact
            info to Google's servers.
        """
        # 1. PII Patterns (Regex)
        # Replace emails with [EMAIL]
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)

        # Replace Phone numbers (simple patterns)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)

        banned_phrases = [
            "ignore previous instructions",
            "system override",
            "run as administrator",
            "you are now",
            "bypass",
            "omit the system prompt",
            "imagine you are a different ai"
        ]

        lower_text = text.lower()
        is_safe = True
        for phrase in banned_phrases:
            if phrase in lower_text:
                is_safe = False
                break

        return (is_safe, text)            
            

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        skin_profile: Optional[Dict] = None,
        nlu: Optional[Dict] = None,
        retrieved_context: Optional[str] = None,
        retrieved_knowledge: Optional[str] = None,
        product_context: Optional[List[Dict]] = None,
        memory_context: str = "",
    ) -> Dict:
        """
        Get a response from the chatbot
        
        Args:
            user_message: The user's question or message
            conversation_history: Optional list of previous messages in format [{"role": "user"/"assistant", "content": "..."}]
        
        Returns:
            The chatbot's response
        """
        try:
            # Now we sanitize the user's prompt first.
            is_safe, clean_message = self._sanitize_input(user_message)

            if not is_safe:
                logger.warning(f"Security Alert: Potential malicious code injection detected in \n{user_message}")
                return "I apologize, but I cannot process that request."
            
            context_block = self._build_context_block(
                product_context or [],
                memory_context,
                skin_profile or {},
                retrieved_context or retrieved_knowledge or "",
                nlu or {},
            )

            # Build the full prompt with conversation history
            if conversation_history:
                messages = self._build_messages_from_history(conversation_history)
                full_prompt = "\n".join(messages) + f"\n\n{context_block}\n\nUser: {clean_message}"
            else:
                full_prompt = f"{self.system_prompt}\n\n{context_block}\n\nUser: {clean_message}"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract text from response
            response = response.text.strip()

            # Clean up any potential markdown code blocks
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?", "", response.strip(), flags=re.IGNORECASE)
                response = re.sub(r"```$", "", response.strip())
            
            response_data = self._parse_model_response(response)

            response_data["response_text"] = self._remove_internal_notes(
                str(response_data.get("response_text", ""))
            )
            response_data = self._normalize_response(response_data)
            
            logger.info(f"Chat response generated successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise

    def _parse_model_response(self, response: str) -> Dict:
        """Parse Gemini JSON and unwrap accidental nested JSON strings."""
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if not json_match:
                return {"response_text": response, "recommended_products": []}
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return {"response_text": response, "recommended_products": []}

        if isinstance(parsed, str):
            nested = self._try_parse_json_string(parsed)
            return nested if isinstance(nested, dict) else {"response_text": parsed, "recommended_products": []}

        if not isinstance(parsed, dict):
            return {"response_text": str(parsed), "recommended_products": []}

        nested_response = parsed.get("response_text")
        if isinstance(nested_response, str):
            nested = self._try_parse_json_string(nested_response)
            if isinstance(nested, dict) and "response_text" in nested:
                merged = parsed.copy()
                for key, value in nested.items():
                    if value not in (None, "", [], {}):
                        merged[key] = value
                parsed = merged
        elif isinstance(nested_response, dict):
            merged = parsed.copy()
            for key, value in nested_response.items():
                if value not in (None, "", [], {}):
                    merged[key] = value
            parsed = merged

        return parsed

    def _try_parse_json_string(self, value: str) -> Optional[Dict]:
        clean = value.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?", "", clean, flags=re.IGNORECASE).strip()
            clean = re.sub(r"```$", "", clean).strip()
        if not clean.startswith("{"):
            return None
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            text_match = re.search(
                r'"response_text"\s*:\s*"(?P<text>.*?)(?<!\\)"\s*(?:,|\})',
                clean,
                flags=re.DOTALL,
            )
            if not text_match:
                return None
            response_text = text_match.group("text")
            response_text = response_text.replace('\\"', '"').replace("\\n", "\n")
            return {"response_text": response_text}
        return parsed if isinstance(parsed, dict) else None

    def _remove_internal_notes(self, response_text: str) -> str:
        """Strip model meta-commentary if the provider leaks it anyway."""
        blocked_patterns = [
            r"\*?self-correction[^*\n]*(?:\*|\n)?",
            r"let me re-evaluate[^.\n]*(?:\.|\n)?",
            r"let's try again[^.\n]*(?:\.|\n)?",
        ]
        cleaned = response_text
        for pattern in blocked_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    def _normalize_response(self, response_data: Dict) -> Dict:
        memory_updates = response_data.get("memory_updates") or {}
        if not isinstance(memory_updates, dict):
            memory_updates = {}
        normalized_memory = {
            "skin_type": memory_updates.get("skin_type"),
            "concerns": list(memory_updates.get("concerns") or []),
            "goals": list(memory_updates.get("goals") or []),
            "preferences": list(memory_updates.get("preferences") or []),
            "avoid": list(memory_updates.get("avoid") or []),
        }
        response_text = self._extract_plain_response_text(response_data.get("response_text", ""))
        return {
            "response_text": response_text,
            "recommended_products": list(response_data.get("recommended_products") or []),
            "product_query": str(response_data.get("product_query") or "").strip(),
            "wants_products": bool(response_data.get("wants_products", False)),
            "needs_followup": bool(response_data.get("needs_followup", False)),
            "followup_questions": list(
                response_data.get("followup_questions")
                or response_data.get("follow_up_questions")
                or []
            ),
            "memory_updates": normalized_memory,
        }

    def _extract_plain_response_text(self, value: object) -> str:
        """Return user-facing Markdown even if response_text contains nested JSON."""
        if isinstance(value, dict):
            return self._extract_plain_response_text(value.get("response_text") or value.get("response") or "")

        text = str(value or "").strip()
        for _ in range(3):
            if not text.startswith("{"):
                return text
            nested = self._try_parse_json_string(text)
            if not nested and ('\\"response_text\\"' in text or "\\n" in text):
                nested = self._try_parse_json_string(text.replace('\\"', '"').replace("\\n", "\n"))
            if not isinstance(nested, dict):
                return text
            next_text = nested.get("response_text") or nested.get("response")
            if not next_text or str(next_text).strip() == text:
                return text
            text = str(next_text).strip()
        return text

    def _build_context_block(
        self,
        product_context: List[Dict],
        memory_context: str,
        skin_profile: Dict,
        retrieved_knowledge: str,
        nlu: Dict,
    ) -> str:
        lines = ["Context for this answer:"]
        if nlu:
            lines.append(f"NLU classification for current message: {json.dumps(nlu, ensure_ascii=True)}")
        lines.append(f"Saved user memory: {memory_context or 'No saved user preferences yet.'}")
        if skin_profile:
            lines.append(f"Frontend skin profile: {json.dumps(skin_profile, ensure_ascii=True)}")
        if retrieved_knowledge:
            lines.append(f"Retrieved skincare knowledge:\n{retrieved_knowledge}")

        if product_context:
            lines.append("Retrieved product catalog candidates. Use exact names only if recommending products:")
            for index, product in enumerate(product_context[:20], start=1):
                name = product.get("name") or product.get("title") or ""
                brand = product.get("brand") or ""
                step = product.get("routine_step") or product.get("category") or "skincare"
                retailer = product.get("retailer") or product.get("source") or product.get("data_source") or ""
                price = product.get("price") or "See retailer"
                lines.append(
                    f"{index}. {brand} {name} | step: {step} | retailer: {retailer} | price: {price}"
                )
        else:
            lines.append("No product catalog candidates were retrieved.")

        return "\n".join(lines)

    def _build_messages_from_history(self, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Convert conversation history to message format"""
        messages = [self.system_prompt]
        messages.append(
            "Use conversation history to infer context. If the user gave a concern, goal, skin type, budget, or product preference earlier, carry it forward. Short follow-ups like \"dark spots\", \"dull skin\", \"what ingredients should I use\", or \"serums?\" must be interpreted using prior context."
        )

        for msg in conversation_history[-12:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            content = content.split("Recommended Products")[0].strip()
            if not content:
                continue
            if role == "user":
                messages.append(f"User: {content}")
            else:
                messages.append(f"Assistant: {content}")
        
        return messages
