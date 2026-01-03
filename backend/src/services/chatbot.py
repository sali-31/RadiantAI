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
        
        self.system_prompt = """You are an expert dermatology assistant and skincare expert. 
            Your role is to provide accurate, helpful information about:
            - Skin conditions and their causes
            - Skincare routines and habits
            - Active ingredients and their benefits
            - Product recommendations and alternatives
            - General dermatological advice
            - Treatment options and when to see a dermatologist

            Guidelines:
            1. Provide evidence-based advice
            2. Always recommend consulting a dermatologist for serious conditions
            3. Be empathetic and encouraging
            4. Explain terms in simple language
            5. Give actionable tips and routines
            6. Consider individual skin types (oily, dry, sensitive, combination, etc.)
            7. Be honest about limitations and when professional help is needed

            Keep responses concise but informative (2-3 paragraphs max).
            Use bullet points for lists when helpful.
            Maintain a friendly, supportive tone.
            
            IMPORTANT: You must return your response in valid JSON format with the following structure:
            {
                "response_text": "Your helpful, empathetic advice here. Use Markdown formatting (bullet points, bold text) for readability.",
                "recommended_products": ["Product 1", "Product 2", ..., "Product N"]
            }

            When recommending products, try to be specific with product names so we can look them up in our product catalog.
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
            

    def chat(self, user_message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
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
            
            # Build the full prompt with conversation history
            if conversation_history:
                messages = self._build_messages_from_history(conversation_history)
                full_prompt = "\n".join(messages) + f"\n\nUser: {clean_message}"
            else:
                full_prompt = f"{self.system_prompt}\n\nUser: {clean_message}"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract text from response
            response = response.text

            # Clean up any potential markdown code blocks
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "")
            
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError:
                # Fallback if AI fails to output JSON
                response_data = {
                    "response_text": response,
                    "recommended_products": []
                }
            
            logger.info(f"Chat response generated successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise

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
