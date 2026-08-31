from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopping_copilot.agent import ShoppingCopilotAgent
from shopping_copilot.config import AgentConfig
from shopping_copilot.features import FEATURE_NAMES
from shopping_copilot.intent import IntentRouter
from shopping_copilot.rerankers import ManualReranker
from shopping_copilot.retrieval import RetrievalResult
from shopping_copilot.slots import SlotExtractor
from shopping_copilot.state import ConversationState


PRODUCTS = [
    {
        "parent_asin": "LEATHER_SHOE",
        "title": "Red Leather Running Shoe",
        "features": ["genuine leather", "red color", "cushioned running comfort"],
        "description": ["lightweight athletic shoe"],
        "price": 79.0,
        "categories": ["Clothing", "Shoes", "Fashion Sneakers"],
        "details": {"department": "womens"},
        "average_rating": 4.8,
        "rating_number": 500,
        "store": "Example Shoes",
    },
    {
        "parent_asin": "BLACK_SHOE",
        "title": "Black Synthetic Fashion Sneaker",
        "features": ["black color", "synthetic upper"],
        "description": ["casual walking shoe"],
        "price": 49.0,
        "categories": ["Clothing", "Shoes", "Fashion Sneakers"],
        "details": {"department": "womens"},
        "average_rating": 4.5,
        "rating_number": 900,
        "store": "Example Shoes",
    },
    {
        "parent_asin": "COTTON_SHIRT",
        "title": "Blue Cotton T-Shirt",
        "features": ["soft cotton", "blue color"],
        "description": ["everyday crew neck top"],
        "price": 20.0,
        "categories": ["Clothing", "Women", "T-Shirts"],
        "details": {"department": "womens"},
        "average_rating": 4.7,
        "rating_number": 1200,
        "store": "Example Apparel",
    },
]


class FrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_intent_router_detects_override(self) -> None:
        route, override = IntentRouter().classify(
            "Actually, ignore my earlier preference. What I need is: leather.",
            "browsing",
        )
        self.assertEqual(route, "buying")
        self.assertTrue(override)

    def test_slot_extractor_tracks_no_preference(self) -> None:
        parsed = SlotExtractor().parse(
            "I don't have an additional preference for material.",
            "browsing",
            "material",
        )
        self.assertEqual(parsed.no_preference, "material")

    def test_override_erases_old_slots_but_keeps_category(self) -> None:
        extractor = SlotExtractor()
        state = ConversationState(session_id="s", user_profile={})
        first = extractor.parse(
            "I'm looking for Shoes Fashion Sneakers. soft cotton lining",
            state.route,
            None,
        )
        state.apply(first, "initial", 1)
        self.assertTrue(state.soft_slots)

        override = extractor.parse(
            "Actually, ignore my earlier preference. What I need is: genuine leather.",
            state.route,
            None,
        )
        state.apply(override, "override", 3)
        self.assertEqual(state.category, "Shoes Fashion Sneakers")
        self.assertFalse(state.soft_slots)
        self.assertIn("material", state.hard_slots)

    def test_idf_weight_prefers_rare_phrases(self) -> None:
        agent = ShoppingCopilotAgent(self.catalog_path)
        index = agent.retriever.index
        self.assertEqual(index.term_weight(""), 0.4)
        self.assertGreaterEqual(index.term_weight("leather"), 0.35)
        self.assertLessEqual(index.term_weight("leather"), 2.5)
        # A term the whole catalog shares must not outweigh a rare, specific one.
        index.idf["ubiquitous"] = 0.0
        index.idf["scarce"] = 8.0
        self.assertLess(index.term_weight("ubiquitous"), index.term_weight("scarce"))

    def test_strong_bm25_hit_is_not_reranked_out(self) -> None:
        agent = ShoppingCopilotAgent(self.catalog_path)
        agent.reset("s", {"preference_tags": [], "summary": ""})
        # Generic constraint that every shoe shares; the target is separated only
        # by the retrieval-rank prior / precise-route seeding.
        response = agent.respond(
            "s", "I'm looking for Shoes Fashion Sneakers. A key requirement is: red leather.", 1, 10
        )
        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertIn("LEATHER_SHOE", identifiers)

    def test_conjunctive_constraint_route_requires_every_term(self) -> None:
        agent = ShoppingCopilotAgent(self.catalog_path)
        rows = agent.retriever.index.search_fts_all("red leather", 10)
        identifiers = [asin for asin, _ in rows]
        self.assertEqual(identifiers[0], "LEATHER_SHOE")
        self.assertNotIn("BLACK_SHOE", identifiers)

    def test_manual_reranker_matches_builtin_score(self) -> None:
        # ManualReranker rescores from the feature vector; its ordering must match
        # the built-in structured score so the model comparison starts from parity.
        message = "I'm looking for Shoes Fashion Sneakers. A key requirement is: genuine leather."

        builtin = ShoppingCopilotAgent(self.catalog_path, AgentConfig(learned_reranker=False))
        self.assertIsNone(builtin.retriever.reranker)
        builtin.reset("s", {"preference_tags": ["material"], "summary": ""})
        base = builtin.respond("s", message, 1, 10)

        ported = ShoppingCopilotAgent(self.catalog_path, AgentConfig(learned_reranker=False))
        ported.retriever.reranker = ManualReranker()
        ported.reset("s", {"preference_tags": ["material"], "summary": ""})
        ported_out = ported.respond("s", message, 1, 10)

        self.assertEqual(
            [r["parent_asin"] for r in base["recommendations"]],
            [r["parent_asin"] for r in ported_out["recommendations"]],
        )

    def test_learned_reranker_loads_and_is_wired(self) -> None:
        from shopping_copilot.learned_reranker import (
            PackagedLambdaRankReranker,
            PackagedLogisticReranker,
        )

        agent = ShoppingCopilotAgent(self.catalog_path)
        self.assertIsInstance(
            agent.retriever.reranker,
            (PackagedLambdaRankReranker, PackagedLogisticReranker),
        )
        # Feature-subset mapping: only the model's own features are consumed.
        self.assertTrue(set(agent.retriever.reranker.feature_names).issubset(set(FEATURE_NAMES)))
        if isinstance(agent.retriever.reranker, PackagedLambdaRankReranker):
            self.assertEqual(agent.retriever.reranker.meta["variant"], "wide")
            self.assertEqual(agent.retriever.config.rerank_depth, 300)
        agent.reset("s", {"preference_tags": [], "summary": ""})
        out = agent.respond("s", "I'm looking for Shoes Fashion Sneakers, but I'm still exploring.", 1, 10)
        self.assertLessEqual(len(out["recommendations"]), 10)

        disabled = ShoppingCopilotAgent(self.catalog_path, AgentConfig(learned_reranker=False))
        self.assertIsNone(disabled.retriever.reranker)

    def test_feature_vector_width_is_stable(self) -> None:
        agent = ShoppingCopilotAgent(self.catalog_path)
        agent.retriever.capture = True
        agent.reset("s", {"preference_tags": [], "summary": ""})
        agent.respond("s", "I'm looking for Shoes Fashion Sneakers, but I'm still exploring.", 1, 10)
        self.assertTrue(agent.retriever.last_candidates)
        for _, vector in agent.retriever.last_candidates:
            self.assertEqual(len(vector), len(FEATURE_NAMES))

    def test_early_pool_promotes_only_a_confident_top_one(self) -> None:
        base = RetrievalResult(
            recommendations=[("BASE_A", 4.0), ("BASE_B", 3.0), ("BASE_C", 2.0)],
            candidate_count=120,
        )
        confident = RetrievalResult(
            recommendations=[("WIDE", 5.0), ("OTHER", 4.3)],
            candidate_count=300,
        )
        promoted = ShoppingCopilotAgent._promote_confident_early_top(
            base, confident, 3, 0.6
        )
        self.assertEqual(
            [asin for asin, _ in promoted.recommendations],
            ["WIDE", "BASE_A", "BASE_B"],
        )
        self.assertEqual(promoted.candidate_count, 300)

        uncertain = RetrievalResult(
            recommendations=[("WIDE", 5.0), ("OTHER", 4.5)],
            candidate_count=300,
        )
        self.assertIs(
            ShoppingCopilotAgent._promote_confident_early_top(base, uncertain, 3, 0.6),
            base,
        )

    def test_agent_returns_contract_compliant_ranked_results(self) -> None:
        agent = ShoppingCopilotAgent(self.catalog_path)
        agent.reset("session", {
            "preference_tags": ["material", "comfort"],
            "summary": "Prefers material and comfort.",
        })
        response = agent.respond(
            "session",
            "I'm looking for Shoes Fashion Sneakers. A key requirement is: genuine leather.",
            1,
            10,
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        })
        self.assertEqual(response["recommendations"][0]["parent_asin"], "LEATHER_SHOE")
        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertLessEqual(len(identifiers), 10)


if __name__ == "__main__":
    unittest.main()
