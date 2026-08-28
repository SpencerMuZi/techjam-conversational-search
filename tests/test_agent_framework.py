from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopping_copilot.agent import ShoppingCopilotAgent
from shopping_copilot.intent import IntentRouter
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
