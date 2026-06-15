# RadiantAI Evaluation

RadiantAI includes two offline evaluation scripts so the MVP can show measurable behavior instead of only demo screenshots.

## Product Recommender

Run:

```bash
.venv/bin/python backend/scripts/eval_recommender.py
```

This compares four strategies:

- `random_products`: a deterministic random sample from the full catalog.
- `top_rated_products`: globally top-rated products, regardless of concern.
- `condition_aware_recommender`: top-rated products from the matching concern catalog.
- `budget_aware_recommender`: matching concern catalog, filtered by budget and balanced across routine categories.

Metrics:

- `top_k_relevance`: fraction of returned products matching the concern tag or concern keywords.
- `budget_fit_rate`: fraction of returned products at or below the scenario budget.
- `category_coverage`: fraction of target routine categories represented.
- `average_rating`: mean catalog rating of returned products.
- `price_diversity`: coefficient of variation for product prices. Higher means more varied price points.

The latest detailed report is written to:

```text
backend/evaluation/recommender_eval_results.json
```

## Chatbot

Run:

```bash
.venv/bin/python backend/scripts/eval_chatbot_local.py
```

The chatbot eval uses 50 skincare prompts from `backend/tests/chatbot_eval_questions.json`. It runs against the local fallback responder so it does not require Gemini credits. Use `--strict` if you want the script to return a failing exit code when cases fail:

```bash
.venv/bin/python backend/scripts/eval_chatbot_local.py --strict
```

Metrics:

- `intent_accuracy`: whether intent detection matches routine/product/ingredient expectations.
- `context_retention`: whether history-based prompts carry earlier skin concerns forward.
- `safety_behavior`: whether medical/safety prompts include appropriate caution and avoid unsafe advice.
- `product_relevance`: whether products appear only when expected and match requested product forms.
- `answer_specificity`: whether answers include expected skincare terms and avoid generic blocked phrases.

The latest detailed report is written to:

```text
backend/evaluation/chatbot_eval_results.json
```
