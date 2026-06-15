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

    def test_brightening_advice_respects_niacinamide_intolerance(self):
        result = main.local_chat_response(
            "I cannot tolerate niacinamide. What can I use for brightening instead?"
        )
        response = result["response"].lower()

        self.assertIn("skip niacinamide", response)
        self.assertIn("tranexamic acid", response)
        self.assertIn("alpha arbutin", response)
        self.assertIn("azelaic acid", response)
        self.assertIn("sunscreen", response)
        self.assertLessEqual(response.count("niacinamide"), 2)
        self.assertNotIn("vitamin c**, **niacinamide", response)
        self.assertEqual(result["products"], [])

    def test_nlu_memory_extracts_niacinamide_intolerance(self):
        nlu = self.get_offline_nlu()
        result = nlu.classify("I cannot tolerate niacinamide. What can I use for brightening instead?")

        self.assertEqual(result["intent"], "ingredient_question")
        self.assertFalse(result["needs_products"])
        self.assertIn("brightening", result["concerns"])
        self.assertIn("niacinamide", result["memory_updates"]["avoid"])

    def test_acne_subtype_answers_are_specific_and_not_repeated(self):
        blackheads = main.local_chat_response("What should I use for blackheads?")["response"].lower()
        whiteheads = main.local_chat_response("What should I use for whiteheads?")["response"].lower()
        forehead = main.local_chat_response("What helps tiny bumps on my forehead?")["response"].lower()

        self.assertIn("blackheads", blackheads)
        self.assertIn("salicylic acid", blackheads)
        self.assertIn("bha", blackheads)

        self.assertIn("whiteheads", whiteheads)
        self.assertIn("closed comedones", whiteheads)
        self.assertIn("adapalene", whiteheads)

        self.assertIn("tiny bumps", forehead)
        self.assertIn("forehead", forehead)
        self.assertTrue("hair" in forehead or "folliculitis" in forehead)

        self.assertNotEqual(blackheads, whiteheads)
        self.assertNotEqual(blackheads, forehead)
        self.assertNotEqual(whiteheads, forehead)

    def test_extended_constraint_answers_respect_avoidance(self):
        no_vitamin_c = main.local_chat_response("I want a dark spot routine without vitamin C.")["response"].lower()
        no_salicylic = main.local_chat_response("I cannot use salicylic acid. What helps blackheads instead?")["response"].lower()
        no_retinol = main.local_chat_response("My skin reacts badly to retinol. What anti-aging ingredients can I use instead?")["response"].lower()
        no_benzoyl = main.local_chat_response("I want an acne routine without benzoyl peroxide.")["response"].lower()

        self.assertNotIn("brightening serum: **vitamin c", no_vitamin_c)
        self.assertIn("tranexamic", no_vitamin_c)
        self.assertIn("adapalene", no_salicylic)
        self.assertNotIn("use a **salicylic acid", no_salicylic)
        self.assertIn("peptides", no_retinol)
        self.assertIn("sunscreen", no_retinol)
        self.assertNotIn("retinoid 2-3 nights", no_retinol)
        self.assertNotIn("use benzoyl peroxide", no_benzoyl)

    def test_current_message_priority_and_no_product_intent(self):
        timing = main.local_chat_response(
            "Ignore my previous dark spot question. How long should I wait between skincare steps?",
            conversation_history=[
                {"role": "user", "content": "Give me products for dark spots"},
                {"role": "assistant", "content": "Use a dark spot routine."},
            ],
        )
        redness = main.local_chat_response("I no longer want product recommendations. Just explain ingredients for redness.")

        self.assertIn("30-60 seconds", timing["response"])
        self.assertEqual(timing["products"], [])
        self.assertIn("azelaic acid", redness["response"].lower())
        self.assertEqual(redness["products"], [])

    def test_application_mixing_sunscreen_and_advanced_answers_are_specific(self):
        mixing = main.local_chat_response("Can I use retinol and benzoyl peroxide together?")["response"].lower()
        sunscreen = main.local_chat_response("How much sunscreen should I use on my face?")
        pie_pih = main.local_chat_response("What is the difference between PIE and PIH acne marks?")["response"].lower()
        pilling = main.local_chat_response("Why does my skincare pill under makeup?")["response"].lower()

        self.assertIn("retinol + benzoyl peroxide", mixing)
        self.assertIn("separate", mixing)
        self.assertIn("two finger", sunscreen["response"].lower())
        self.assertEqual(sunscreen["products"], [])
        self.assertIn("post-inflammatory erythema", pie_pih)
        self.assertIn("post-inflammatory hyperpigmentation", pie_pih)
        self.assertIn("pilling", pilling)
        self.assertNotIn("share your skin type", pilling)

    def test_product_and_body_answers_are_specific(self):
        moisturizer = main.local_chat_response("I am allergic to fragrance. Recommend a moisturizer.")
        sunscreen = main.local_chat_response("Recommend only sunscreens for oily skin.")
        back_acne = main.local_chat_response("What helps back acne?")["response"].lower()

        self.assertIn("fragrance", moisturizer["response"].lower())
        self.assertGreater(len(moisturizer["products"]), 0)
        self.assertIn("sunscreen recommendations", sunscreen["response"].lower())
        self.assertGreater(len(sunscreen["products"]), 0)
        self.assertIn("body acne", back_acne)
        self.assertIn("benzoyl peroxide wash", back_acne)


if __name__ == "__main__":
    unittest.main()
