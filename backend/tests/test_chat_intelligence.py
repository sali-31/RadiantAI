import os
import tempfile
import unittest
from pathlib import Path


os.environ["ENABLE_LIVE_PRODUCT_SEARCH"] = "false"

from backend.src import main  # noqa: E402
from backend.src.services import memory as memory_service  # noqa: E402
from backend.src.services.skincare_nlu import SkincareNLU  # noqa: E402
from backend.src.services.skincare_rag import SkincareRAG  # noqa: E402


class ChatIntelligenceTests(unittest.TestCase):
    def get_offline_nlu(self):
        nlu = SkincareNLU()
        nlu.model = None
        return nlu

    def test_nlu_classifies_closed_comedones_as_acne_current_message(self):
        nlu = self.get_offline_nlu()
        result = nlu.classify(
            "skincare routine for closed comedones",
            conversation_history=[
                {"role": "user", "content": "routine for anti aging"},
                {"role": "assistant", "content": "Here is an anti-aging routine"},
            ],
        )

        self.assertEqual(result["intent"], "routine_request")
        self.assertIn("closed_comedones", result["concerns"])
        self.assertIn("acne", result["concerns"])
        self.assertNotIn("anti_aging", result["concerns"])

    def test_nlu_classifies_application_timing_without_products(self):
        nlu = self.get_offline_nlu()
        result = nlu.classify("How long should I wait between skincare steps?")

        self.assertEqual(result["intent"], "application_timing")
        self.assertEqual(result["answer_type"], "timing")
        self.assertFalse(result["needs_products"])

    def test_nlu_classifies_brightening_ingredient_and_serum_requests(self):
        nlu = self.get_offline_nlu()
        ingredients = nlu.classify("skincare ingredients for brightening")
        serums = nlu.classify("give me serum recommendations for brightening")

        self.assertEqual(ingredients["intent"], "ingredient_question")
        self.assertFalse(ingredients["needs_products"])
        self.assertEqual(serums["intent"], "product_recommendation")
        self.assertEqual(serums["product_form"], "serum")
        self.assertTrue(serums["needs_products"])

    def test_rag_retrieves_application_timing_chunk(self):
        nlu = self.get_offline_nlu().classify("How long should I wait between skincare steps?")
        chunks = SkincareRAG().retrieve("How long should I wait between skincare steps?", nlu, top_k=2)

        self.assertEqual(chunks[0]["id"], "application_timing")

    def test_local_chat_timing_does_not_return_acne_routine(self):
        response = main.local_chat_response("How long should I wait between skincare steps?")["response"]

        self.assertIn("30-60 seconds", response)
        self.assertNotIn("acne-focused routine", response.lower())

    def test_current_message_overrides_old_anti_aging_context(self):
        response = main.local_chat_response(
            "skincare routine for closed comedones",
            conversation_history=[
                {"role": "user", "content": "routine for anti aging"},
                {"role": "assistant", "content": "Here is an anti-aging routine"},
            ],
        )["response"]

        self.assertIn("closed-comedone routine", response)
        self.assertNotIn("anti-aging routine", response.lower())

    def test_kojic_serum_query_returns_exact_grounded_products(self):
        products = main.products_for_chat_message(
            "give me good kojic acid serums",
            "hyperpigmentation",
            limit=8,
        )

        self.assertGreaterEqual(len(products), 2)
        first_name = f"{products[0].get('title', '')} {products[0].get('name', '')}".lower()
        self.assertIn("kojic", first_name)
        self.assertIn("serum", first_name)

    def test_routine_query_returns_two_products_for_each_core_step(self):
        products = main.products_for_chat_message(
            "build me a korean routine for oily acne prone skin under $60",
            "acne",
            limit=16,
        )
        counts = {}
        for product in products:
            step = product.get("routine_step") or product.get("category")
            counts[step] = counts.get(step, 0) + 1

        for step in ["cleanser", "toner", "treatment", "moisturizer", "sunscreen"]:
            self.assertGreaterEqual(counts.get(step, 0), 2, f"missing products for {step}")

    def test_memory_extracts_user_preferences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = memory_service.MEMORY_PATH
            memory_service.MEMORY_PATH = Path(temp_dir) / "chat_memory.json"
            try:
                updated = memory_service.update_user_memory(
                    "test-user",
                    "I have oily acne-prone skin, prefer Anua and COSRX, avoid fragrance, budget under $50",
                    [],
                )
            finally:
                memory_service.MEMORY_PATH = original_path

        self.assertEqual(updated["skin_type"], "acne-prone")
        self.assertIn("acne", updated["concerns"])
        self.assertIn("fragrance", updated["avoided_ingredients"])
        self.assertEqual(updated["budget_max"], 50)
        self.assertIn("anua", updated["preferred_brands"])
        self.assertIn("cosrx", updated["preferred_brands"])


if __name__ == "__main__":
    unittest.main()
