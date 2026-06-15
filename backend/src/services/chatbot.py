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
        # Updated to use a supported model
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        self.system_prompt = """You are RadiantAI, a careful skincare advisor for a consumer skincare app.

            Your job:
            - Give practical skincare guidance in clear language.
            - Use the provided user memory when it is relevant.
            - Use the provided product catalog context as the source of truth for product recommendations.
            - Prefer products from Sephora, StyleKorean, popular Korean brands, and the local verified/Amazon catalog when provided.
            - Ask 1-3 focused follow-up questions only when the answer would be unsafe or too vague without them.

            Safety and quality rules:
            1. Do not diagnose. Say when a dermatologist is needed for painful, spreading, infected, severe, scarring, or persistent symptoms.
            2. Avoid weak generic answers. If products are requested, give concrete product/routine guidance.
            3. Respect allergies, sensitivities, budget, skin type, and avoided ingredients from memory.
            4. Never invent retailer links, prices, ratings, or product availability.
            5. If catalog products are provided, recommend only those catalog products by exact name.
            6. If the user asks for a routine, cover cleanser, toner, treatment, moisturizer, and sunscreen.
            7. For routines, include at least 2 product options for each step when the catalog context supports it.
            8. Keep active ingredients realistic: introduce strong actives slowly and remind users to use SPF with brightening acids or retinoids.
            9. Do not reveal internal reasoning, drafting notes, uncertainty narration, or self-corrections.
            10. Never write phrases like "self-correction", "let me re-evaluate", "I should", or "try again".

            Return valid JSON only:
            {
                "response_text": "Concise Markdown answer. Use headings or bullets when helpful.",
                "recommended_products": ["Exact Product Name 1", "Exact Product Name 2"],
                "follow_up_questions": ["Question 1"],
                "remembered_facts": {"skin_type": "", "budget_max": null, "concerns": [], "allergies": [], "preferred_brands": []}
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
            
            context_block = self._build_context_block(product_context or [], memory_context)

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
            
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
                if json_match:
                    try:
                        response_data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        response_data = {
                            "response_text": response,
                            "recommended_products": []
                        }
                else:
                    response_data = {
                        "response_text": response,
                        "recommended_products": []
                    }

            response_data["response_text"] = self._remove_internal_notes(
                str(response_data.get("response_text", ""))
            )
            response_data.setdefault("recommended_products", [])
            response_data.setdefault("follow_up_questions", [])
            response_data.setdefault("remembered_facts", {})
            
            logger.info(f"Chat response generated successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise

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

    def _build_context_block(self, product_context: List[Dict], memory_context: str) -> str:
        lines = ["Context for this answer:"]
        lines.append(f"Saved user memory: {memory_context or 'No saved user preferences yet.'}")

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
        
        for msg in conversation_history[-5:]:  # Keep last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(f"User: {content}")
            else:
                messages.append(f"Assistant: {content}")
        
        return messages
