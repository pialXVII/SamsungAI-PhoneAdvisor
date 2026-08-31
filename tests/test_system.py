"""Test suite for the Samsung Phone Query and Review System.

Run with:  python tests/test_system.py

Uses only the standard library so the suite runs without extra installs. The
LLM is disabled throughout: these tests check the deterministic machinery —
parsing, storage, retrieval, routing and the API contract — not the wording a
generative model happens to produce.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Must be set before config is imported anywhere.
os.environ["USE_LLM"] = "false"

from src.database.db import session_scope
from src.database.repository import (
    count_phones,
    extract_mentioned_phones,
    find_phone,
    get_all_phones,
    top_by_column,
)
from src.rag.query_analysis import Intent, analyze, detect_aspects
from src.scraper.parsers import (
    parse_battery_mah,
    parse_camera_mp,
    parse_charging_watts,
    parse_display_inches,
    parse_memory_options,
    parse_prices,
    parse_refresh_rate,
    parse_release_year,
    parse_weight_grams,
)


class TestParsers(unittest.TestCase):
    """The regex layer that turns GSMArena prose into numbers."""

    def test_battery(self):
        self.assertEqual(parse_battery_mah("Li-Ion 5000 mAh, non-removable"), 5000)
        self.assertEqual(parse_battery_mah("Li-Ion 3900 mAh"), 3900)
        self.assertIsNone(parse_battery_mah("Removable battery"))
        self.assertIsNone(parse_battery_mah(None))

    def test_display_inches(self):
        self.assertEqual(
            parse_display_inches("6.1 inches, 90.1 cm2 (~86.8% screen-to-body ratio)"),
            6.1,
        )
        self.assertEqual(parse_display_inches("7.6 inches"), 7.6)
        self.assertIsNone(parse_display_inches("Foldable display"))

    def test_refresh_rate_takes_the_maximum(self):
        self.assertEqual(
            parse_refresh_rate("Dynamic AMOLED 2X, 120Hz, HDR10+, 1750 nits"), 120
        )
        self.assertEqual(parse_refresh_rate("LTPO AMOLED, 1-120Hz"), 120)
        self.assertIsNone(parse_refresh_rate("Super AMOLED"))

    def test_camera_takes_the_primary_sensor_not_the_largest(self):
        # The headline sensor is the one listed first, which is not always the
        # highest number in the string.
        self.assertEqual(parse_camera_mp("50 MP, f/1.8 wide; 10 MP telephoto"), 50.0)
        self.assertEqual(parse_camera_mp("200 MP, f/1.7, 24mm (wide)"), 200.0)
        self.assertIsNone(parse_camera_mp("No camera"))

    def test_memory_separates_ram_from_storage(self):
        ram, storage = parse_memory_options("128GB 8GB RAM, 256GB 12GB RAM")
        self.assertEqual((ram, storage), (12, 256))

        ram, storage = parse_memory_options("1TB 12GB RAM")
        self.assertEqual((ram, storage), (12, 1024))

        self.assertEqual(parse_memory_options(None), (None, None))

    def test_charging_watts(self):
        self.assertEqual(
            parse_charging_watts("45W wired, 15W wireless, 4.5W reverse"), 45.0
        )
        self.assertIsNone(parse_charging_watts("Fast charging"))

    def test_weight_and_year(self):
        self.assertEqual(parse_weight_grams("168 g (5.93 oz)"), 168.0)
        self.assertEqual(parse_release_year("2023, February 01"), 2023)
        self.assertIsNone(parse_release_year("Coming soon"))

    def test_prices_split_by_currency(self):
        prices = parse_prices("$ 434.99 / € 441.44 / £ 424.99 / ₹ 89,999")
        self.assertEqual(len(prices), 4)
        self.assertEqual(prices[0], {
            "currency": "USD", "amount": 434.99,
            "raw_text": "$ 434.99", "source": "GSMArena",
        })
        # Thousands separators must not truncate the value.
        self.assertEqual(prices[3]["amount"], 89999.0)
        self.assertEqual(parse_prices(None), [])


class TestQueryAnalysis(unittest.TestCase):
    """Intent routing — what decides how a question gets answered."""

    def test_aspect_detection(self):
        self.assertEqual(detect_aspects("camera specs of the S23")[0], "camera")
        self.assertEqual(detect_aspects("which has the best battery life")[0], "battery")
        self.assertEqual(detect_aspects("how big is the screen")[0], "display")

    def test_superlative_routes_to_ranking(self):
        analysis = analyze("Which Samsung phone has the best battery life?")
        self.assertEqual(analysis.intent, Intent.SUPERLATIVE)
        self.assertEqual(analysis.ranking_column, "battery_capacity_mah")
        self.assertTrue(analysis.higher_is_better)

    def test_two_named_phones_always_means_comparison(self):
        # Even with no comparison keyword at all.
        analysis = analyze("S23 or S22 for gaming?", mentioned_phone_count=2)
        self.assertEqual(analysis.intent, Intent.COMPARISON)

    def test_single_phone_is_a_spec_lookup(self):
        analysis = analyze("camera specs of the S23", mentioned_phone_count=1)
        self.assertEqual(analysis.intent, Intent.SPEC_LOOKUP)

    def test_lightest_ranks_ascending(self):
        analysis = analyze("Which is the lightest phone?")
        self.assertEqual(analysis.intent, Intent.SUPERLATIVE)
        self.assertFalse(analysis.higher_is_better)


class TestDatabase(unittest.TestCase):
    """Storage and lookup against the live scraped database."""

    @classmethod
    def setUpClass(cls):
        with session_scope() as session:
            if count_phones(session) == 0:
                raise unittest.SkipTest(
                    "Database is empty — run `python scripts/scrape.py` first"
                )

    def test_scrape_covered_the_target_range(self):
        with session_scope() as session:
            self.assertGreaterEqual(count_phones(session), 10)

    def test_core_fields_are_populated(self):
        with session_scope() as session:
            for phone in get_all_phones(session):
                self.assertTrue(phone.chipset, f"{phone.name} has no chipset")
                self.assertIsNotNone(
                    phone.battery_capacity_mah, f"{phone.name} has no battery"
                )
                self.assertIsNotNone(
                    phone.display_size_inches, f"{phone.name} has no display size"
                )
                self.assertGreater(len(phone.specifications), 20)

    def test_fuzzy_lookup_prefers_the_more_specific_model(self):
        with session_scope() as session:
            ultra = find_phone(session, "s23 ultra")
            self.assertIsNotNone(ultra)
            self.assertIn("Ultra", ultra.name)

            base = find_phone(session, "galaxy s23")
            self.assertIsNotNone(base)
            self.assertNotIn("Ultra", base.name)

    def test_comparison_questions_resolve_both_models(self):
        with session_scope() as session:
            phones = extract_mentioned_phones(
                session, "How does the Galaxy S23 compare to the S22 in performance?"
            )
            names = {p.name for p in phones}
            self.assertEqual(len(phones), 2, f"expected 2 phones, got {names}")
            self.assertTrue(any("S23" in n for n in names))
            self.assertTrue(any("S22" in n for n in names))

    def test_ranking_is_correctly_ordered(self):
        with session_scope() as session:
            ranked = top_by_column(session, "battery_capacity_mah", limit=5)
            capacities = [p.battery_capacity_mah for p in ranked]
            self.assertEqual(capacities, sorted(capacities, reverse=True))


class TestRAG(unittest.TestCase):
    """Document building and semantic retrieval."""

    @classmethod
    def setUpClass(cls):
        with session_scope() as session:
            if count_phones(session) == 0:
                raise unittest.SkipTest("Database is empty")

        from src.rag.chatbot import SamsungChatbot

        cls.bot = SamsungChatbot(use_llm=False)
        cls.bot.prepare()

    def test_corpus_has_multiple_aspects_per_phone(self):
        stats = self.bot.vector_store.stats()
        self.assertTrue(stats["ready"])
        with session_scope() as session:
            phone_count = count_phones(session)
        self.assertGreater(stats["documents"], phone_count * 3)

    def test_spec_lookup_retrieves_only_the_asked_about_aspect(self):
        response = self.bot.chat("What are the camera specs of the Galaxy S23?")
        self.assertEqual(response.intent, "spec_lookup")
        self.assertTrue(response.sources)
        # This is the regression guard for display specs leaking into a camera
        # answer, which made the model report the screen's refresh rate as a
        # selfie-camera capability.
        self.assertTrue(
            all(s["aspect"] == "Camera" for s in response.sources),
            f"off-topic passages retrieved: {response.sources}",
        )

    def test_superlative_answer_names_the_actual_winner(self):
        response = self.bot.chat("Which Samsung phone has the best battery life?")
        self.assertEqual(response.intent, "superlative")
        with session_scope() as session:
            best = top_by_column(session, "battery_capacity_mah", limit=1)[0]
        self.assertIn(best.name, response.answer)

    def test_comparison_pulls_in_both_phones(self):
        response = self.bot.chat(
            "How does the Galaxy S23 compare to the S22 in terms of performance?"
        )
        self.assertEqual(response.intent, "comparison")
        self.assertEqual(len(response.phones), 2)

    def test_unknown_topic_is_declined_rather_than_guessed(self):
        response = self.bot.chat("What is the price of a Toyota Corolla?")
        self.assertTrue(response.answer)


class TestAgentBackends(unittest.TestCase):
    """Selection between the CrewAI and native orchestrators."""

    def test_native_is_selected_when_forced(self):
        import config
        from src.agents import crew

        original = config.AGENT_FRAMEWORK
        try:
            config.AGENT_FRAMEWORK = "native"
            self.assertEqual(crew.active_backend(), "native")
        finally:
            config.AGENT_FRAMEWORK = original

    def test_crewai_unavailable_falls_back_rather_than_raising(self):
        # USE_LLM=false in this suite, so CrewAI reports unavailable (its agents
        # have no template fallback) and selection must degrade quietly.
        from src.agents import crew
        from src.agents import crewai_backend

        self.assertFalse(crewai_backend.is_available())
        self.assertEqual(crew.active_backend(), "native")

    def test_backend_status_is_reportable(self):
        from src.agents.crew import backend_status

        status = backend_status()
        self.assertIn(status["active"], ("crewai", "native"))
        self.assertIn("crewai_installed", status)


class TestAgents(unittest.TestCase):
    """Multi-agent workflows."""

    @classmethod
    def setUpClass(cls):
        with session_scope() as session:
            if count_phones(session) == 0:
                raise unittest.SkipTest("Database is empty")

    def test_review_crew_runs_both_agents_in_order(self):
        from src.agents.crew import generate_review

        result = generate_review("Galaxy S23 Ultra")
        self.assertTrue(result["success"])
        self.assertIn("Ultra", result["phone"])
        self.assertEqual(
            list(result["agents"].keys()), ["specifications", "review"]
        )
        self.assertEqual(
            result["agents"]["specifications"]["agent"], "SpecificationAgent"
        )
        self.assertEqual(result["framework"], "native")
        self.assertTrue(result["review"])

    def test_review_uses_real_scraped_numbers(self):
        from src.agents.crew import generate_review

        result = generate_review("Galaxy S23 Ultra")
        numbers = result["key_numbers"]
        self.assertEqual(numbers["battery_mah"], 5000)
        self.assertEqual(numbers["main_camera_mp"], 200.0)

    def test_unknown_model_fails_cleanly(self):
        from src.agents.crew import generate_review

        result = generate_review("Nokia 3310")
        self.assertFalse(result["success"])
        self.assertIn("Available models", result["review"])

    def test_comparison_crew_builds_a_difference_table(self):
        from src.agents.crew import compare_phones

        result = compare_phones("Galaxy S23", "Galaxy S22", focus="performance")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["phones"]), 2)
        self.assertIn("Chipset", result["table"])
        self.assertIn("Battery", result["table"])

    def test_free_text_routes_to_the_right_crew(self):
        from src.agents.crew import handle_request

        self.assertEqual(
            handle_request("Review the Galaxy S24 Ultra")["workflow"], "review"
        )
        self.assertEqual(
            handle_request("Galaxy S23 vs Galaxy S22 camera")["workflow"], "comparison"
        )
        self.assertFalse(handle_request("Tell me about cars")["success"])


class TestAPI(unittest.TestCase):
    """HTTP contract, exercised in-process with FastAPI's TestClient."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # httpx is a TestClient dependency
            raise unittest.SkipTest(f"TestClient unavailable: {exc}")

        with session_scope() as session:
            if count_phones(session) == 0:
                raise unittest.SkipTest("Database is empty")

        from src.api.main import app

        cls.client = TestClient(app)

    def test_health_reports_all_subsystems(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "healthy")
        self.assertTrue(body["database"]["connected"])
        self.assertGreater(body["database"]["phones"], 0)
        self.assertTrue(body["vector_store"]["ready"])

    def test_list_and_detail(self):
        phones = self.client.get("/phones").json()
        self.assertGreaterEqual(len(phones), 10)

        detail = self.client.get(f"/phones/{phones[0]['id']}").json()
        self.assertEqual(detail["name"], phones[0]["name"])
        self.assertTrue(detail["specifications"])

    def test_search_route_is_not_shadowed_by_the_id_route(self):
        # /phones/search must not be parsed as /phones/{phone_id}.
        response = self.client.get("/phones/search", params={"q": "s23 ultra"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json())

    def test_chat_endpoint(self):
        body = self.client.post(
            "/chat", json={"query": "What is the screen size of the Galaxy S22?"}
        ).json()
        self.assertTrue(body["answer"])
        self.assertEqual(body["intent"], "spec_lookup")

    def test_agent_endpoints(self):
        review = self.client.post(
            "/agents/review", json={"phone": "Galaxy S23 Ultra"}
        ).json()
        self.assertTrue(review["success"])

        comparison = self.client.post(
            "/agents/compare",
            json={"phone_a": "Galaxy S23", "phone_b": "Galaxy S22", "focus": "camera"},
        ).json()
        self.assertTrue(comparison["success"])

    def test_error_paths(self):
        self.assertEqual(self.client.get("/phones/99999").status_code, 404)
        self.assertEqual(
            self.client.post("/agents/review", json={"phone": "Nokia 3310"}).status_code,
            404,
        )
        # Empty query violates the min_length constraint.
        self.assertEqual(self.client.post("/chat", json={"query": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
