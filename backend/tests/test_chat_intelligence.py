import os
import tempfile
import unittest
from pathlib import Path


os.environ["ENABLE_LIVE_PRODUCT_SEARCH"] = "false"

from backend.src import main  # noqa: E402
from backend.src.services import memory as memory_service  # noqa: E402


class ChatIntelligenceTests(unittest.TestCase):
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
