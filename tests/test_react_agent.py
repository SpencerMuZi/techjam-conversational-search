from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from shopping_copilot.agent import ShoppingCopilotAgent
from shopping_copilot.llm import DeepSeekClient, LLMConfig, LLMError, LLMReply
from tests.test_agent_framework import PRODUCTS


def action(name, **arguments):
    return {"id": "call_" + name, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
    }}


class ScriptedClient:
    """Actions may depend on observations, rather than only a fixed sequence."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.messages = []

    def complete(self, messages, tools):
        self.messages.append(copy.deepcopy(messages))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        calls = step(messages) if callable(step) else step
        return LLMReply({"role": "assistant", "content": "", "tool_calls": calls}, 100, 20)


class ReActTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = Path(self.directory.name) / "catalog.jsonl"
        self.catalog.write_text("".join(json.dumps(p) + "\n" for p in PRODUCTS), encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def agent(self, client, **options):
        agent = ShoppingCopilotAgent(self.catalog, mode="react", llm_client=client,
                                     llm_config=LLMConfig(**options))
        agent.reset("s", {"preference_tags": [], "summary": ""})
        return agent

    def test_model_selects_details_and_final_recommendation_from_observation(self):
        def inspect(messages):
            observation = json.loads(messages[-1]["content"])
            self.assertIn("products", observation)
            chosen = next(p["parent_asin"] for p in observation["products"] if "leather" in p["features"])
            return [action("get_product_details", parent_asins=[chosen])]

        def finish(messages):
            product = json.loads(messages[-1]["content"])["products"][0]
            self.assertEqual(product["price"], 79)
            return [action("finish", message="这款红色真皮鞋符合你的材质要求。", parent_asins=[product["parent_asin"]])]

        client = ScriptedClient(
            [action("update_search_state", route="buying", category="Shoes Fashion Sneakers",
                    hard_slots={"material": ["genuine leather"], "color": ["red"]})],
            [action("search_products", query="red genuine leather shoes")], inspect, finish,
        )
        agent = self.agent(client)
        response = agent.respond("s", "帮我找红色真皮鞋", 1, 1)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "LEATHER_SHOE")
        self.assertEqual(response["usage"], {"prompt_tokens": 400, "completion_tokens": 80})
        self.assertEqual([event["tool"] for event in agent.last_trace],
                         ["update_search_state", "search_products", "get_product_details", "finish"])
        self.assertEqual(agent.sessions.get("s").hard_slots["material"], ["genuine leather"])
        self.assertEqual(len(agent.sessions.get("s").history), 2)
        self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})

    def test_clarification_can_precede_search_and_followup_keeps_context(self):
        client = ScriptedClient(
            [action("update_search_state", category="Shoes", soft_slots={"material": ["leather"]})],
            [action("ask_user", attribute="color", question="你喜欢什么颜色？")],
            [action("update_search_state", soft_slots={"color": ["red"]})],
            [action("search_products", query="red leather shoes")],
            [action("finish", message="推荐这双鞋。", parent_asins=["LEATHER_SHOE"])],
        )
        agent = self.agent(client)
        first = agent.respond("s", "想看看真皮鞋", 1, 10)
        self.assertEqual(first["ask_attribute"], "color")
        self.assertEqual(first["recommendations"], [])
        agent.respond("s", "红色", 2, 10)
        state = agent.sessions.get("s")
        self.assertEqual(state.soft_slots, {"material": ["leather"], "color": ["red"]})
        self.assertEqual(len(state.history), 4)
        self.assertTrue(any(m["content"] == "你喜欢什么颜色？" for m in client.messages[2]))

    def test_repeated_search_is_chosen_by_model(self):
        client = ScriptedClient(
            [action("search_products", query="red leather")],
            [action("search_products", query="black synthetic sneakers")],
            [action("finish", message="另一款候选。", parent_asins=["BLACK_SHOE"])],
        )
        agent = self.agent(client)
        response = agent.respond("s", "比较两类鞋", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "BLACK_SHOE")
        self.assertEqual([e["tool"] for e in agent.last_trace].count("search_products"), 2)

    def test_invalid_arguments_and_hallucinated_ids_are_repairable(self):
        malformed = action("search_products", query="shoes")
        malformed["function"]["arguments"] = "{invalid"
        client = ScriptedClient(
            [malformed], [action("finish", message="fake", parent_asins=["NONEXISTENT"])],
            [action("get_product_details", parent_asins=["LEATHER_SHOE"])],
            [action("unknown_tool")],
            [action("ask_user", attribute="category", question="想找什么商品？")],
        )
        agent = self.agent(client)
        response = agent.respond("s", "帮我挑选", 1, 10)
        self.assertEqual(response["ask_attribute"], "category")
        self.assertTrue(all("error" in e["observation"] for e in agent.last_trace[:4]))

    def test_state_corrections_clear_stale_requirements_and_candidates(self):
        client = ScriptedClient(
            [action("update_search_state", category="Shoes", hard_slots={"color": ["red"]})],
            [action("search_products", query="red leather shoes")],
            [action("update_search_state", hard_slots={"color": ["black"]})],
            [action("finish", message="stale", parent_asins=["LEATHER_SHOE"])],
            [action("search_products", query="black synthetic shoes")],
            [action("finish", message="改为黑色。", parent_asins=["BLACK_SHOE"])],
            [action("update_search_state", category="T-Shirts", clear=True,
                    hard_slots={"material": ["cotton"]})],
            [action("ask_user", attribute="size", question="上衣需要什么尺码？")],
        )
        agent = self.agent(client)
        agent.respond("s", "不要红色，改黑色", 1, 10)
        self.assertIn("error", agent.last_trace[3]["observation"])
        self.assertEqual(agent.sessions.get("s").hard_slots["color"], ["black"])
        agent.respond("s", "改买棉上衣", 2, 10)
        self.assertEqual(agent.sessions.get("s").hard_slots, {"material": ["cotton"]})

    def test_no_preference_removes_constraint_and_cannot_be_asked_again(self):
        client = ScriptedClient(
            [action("update_search_state", hard_slots={"color": ["red"]})],
            [action("update_search_state", no_preference=["color"])],
            [action("ask_user", attribute="color", question="颜色？")],
            [action("ask_user", attribute="budget", question="预算是多少？")],
        )
        agent = self.agent(client)
        response = agent.respond("s", "颜色无所谓", 1, 10)
        self.assertEqual(response["ask_attribute"], "budget")
        self.assertFalse(agent.sessions.get("s").hard_slots)
        self.assertIn("error", agent.last_trace[2]["observation"])

    def test_provider_failure_rolls_back_state_and_fallback_records_turn_once(self):
        client = ScriptedClient(
            [action("update_search_state", category="Bogus", hard_slots={"color": ["pink"]})],
            LLMError("LLM HTTP 429"),
        )
        agent = self.agent(client)
        response = agent.respond("s", "I'm looking for Shoes. A key requirement is: leather.", 1, 10)
        state = agent.sessions.get("s")
        self.assertEqual(state.category, "Shoes")
        self.assertNotIn("color", state.hard_slots)
        self.assertEqual(len(state.history), 2)
        self.assertEqual(response["usage"]["prompt_tokens"], 100)
        self.assertEqual(agent.last_error, "LLM HTTP 429")

    def test_strict_mode_step_limit_raises_without_committing(self):
        client = ScriptedClient([action("update_search_state", category="Shoes")])
        agent = self.agent(client, max_steps=1, fallback=False)
        with self.assertRaisesRegex(LLMError, "step limit"):
            agent.respond("s", "shoes", 1, 10)
        self.assertFalse(agent.sessions.get("s").history)
        self.assertEqual(agent.sessions.get("s").category, "")

    def test_terminal_batch_is_rejected_and_every_call_gets_an_observation(self):
        client = ScriptedClient(
            [action("finish", message="premature", parent_asins=[]), action("search_products", query="shoes")],
            [action("finish", message="完成", parent_asins=["LEATHER_SHOE"])],
        )
        agent = self.agent(client)
        agent.respond("s", "shoes", 1, 10)
        self.assertIn("error", agent.last_trace[0]["observation"])
        tools = [m for m in client.messages[1] if m["role"] == "tool"]
        self.assertEqual(len(tools), 2)

    def test_duplicate_ids_and_top_k_are_enforced(self):
        client = ScriptedClient(
            [action("search_products", query="shoes")],
            [action("finish", message="dupes", parent_asins=["LEATHER_SHOE", "LEATHER_SHOE"])],
            [action("finish", message="too many", parent_asins=["LEATHER_SHOE", "BLACK_SHOE"])],
            [action("finish", message="一款", parent_asins=["LEATHER_SHOE"])],
        )
        agent = self.agent(client)
        self.assertEqual(len(agent.respond("s", "shoes", 1, 1)["recommendations"]), 1)
        self.assertIn("error", agent.last_trace[1]["observation"])
        self.assertIn("error", agent.last_trace[2]["observation"])

    def test_reset_removes_previous_context_and_recommendations(self):
        client = ScriptedClient([action("search_products", query="shoes")],
                                [action("finish", message="shoe", parent_asins=["LEATHER_SHOE"])])
        agent = self.agent(client)
        agent.respond("s", "shoes", 1, 10)
        agent.reset("s", {})
        self.assertFalse(agent.sessions.get("s").last_recommendations)
        self.assertFalse(agent.sessions.get("s").history)

    def test_missing_key_falls_back_without_network_and_offline_never_calls_llm(self):
        with patch("shopping_copilot.llm.urlopen") as request:
            agent = ShoppingCopilotAgent(self.catalog, mode="react", llm_config=LLMConfig())
            agent.reset("s", {})
            agent.respond("s", "I'm looking for Shoes.", 1, 10)
            self.assertIn("not configured", agent.last_error)
            request.assert_not_called()
        client = ScriptedClient()
        agent = ShoppingCopilotAgent(self.catalog, mode="offline", llm_client=client)
        agent.reset("s", {})
        self.assertEqual(agent.respond("s", "shoes", 1, 10)["usage"]["prompt_tokens"], 0)
        self.assertFalse(client.messages)

    def test_hard_budget_rejects_unknown_and_excess_prices_before_finish(self):
        from dataclasses import replace
        client = ScriptedClient(
            [action("update_search_state", hard_slots={"budget": ["under 50"]}, max_price=50)],
            [action("search_products", query="shoes cotton")],
            [action("finish", message="unknown", parent_asins=["COTTON_SHIRT"])],
            [action("finish", message="expensive", parent_asins=["LEATHER_SHOE"])],
            [action("finish", message="within budget", parent_asins=["BLACK_SHOE"])],
        )
        agent = self.agent(client)
        index = agent.retriever.index
        index.documents["COTTON_SHIRT"] = replace(index.documents["COTTON_SHIRT"], price=None)
        response = agent.respond("s", "预算50以内", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "BLACK_SHOE")
        self.assertIn("unknown price", agent.last_trace[2]["observation"]["error"])
        self.assertIn("exceeds max_price", agent.last_trace[3]["observation"]["error"])

    def test_hard_budget_needs_explicit_numeric_ceiling(self):
        client = ScriptedClient(
            [action("update_search_state", hard_slots={"budget": ["under 50"]})],
            [action("search_products", query="shoes")],
            [action("finish", message="missing ceiling", parent_asins=["BLACK_SHOE"])],
            [action("update_search_state", max_price=50)],
            [action("search_products", query="shoes")],
            [action("finish", message="within budget", parent_asins=["BLACK_SHOE"])],
        )
        agent = self.agent(client)
        agent.respond("s", "预算50以内", 1, 10)
        self.assertIn("Set max_price", agent.last_trace[2]["observation"]["error"])

    def test_budget_changes_reset_and_no_preference_clear_ceiling(self):
        client = ScriptedClient(
            [action("update_search_state", category="Shoes", hard_slots={"budget": ["under 100"]}, max_price=100)],
            [action("ask_user", attribute="color", question="颜色？")],
            [action("update_search_state", hard_slots={"budget": ["under 50"]}, max_price=50)],
            [action("ask_user", attribute="size", question="尺码？")],
            [action("update_search_state", no_preference=["budget"])],
            [action("ask_user", attribute="brand", question="品牌？")],
            [action("update_search_state", max_price=80)],
            [action("update_search_state", category="T-Shirts", clear=True)],
            [action("ask_user", attribute="size", question="上衣尺码？")],
        )
        agent = self.agent(client)
        agent.respond("s", "预算100", 1, 10)
        self.assertEqual(agent.sessions.get("s").max_price, 100)
        agent.respond("s", "预算改50", 2, 10)
        self.assertEqual(agent.sessions.get("s").max_price, 50)
        agent.respond("s", "预算无所谓", 3, 10)
        self.assertIsNone(agent.sessions.get("s").max_price)
        agent.respond("s", "换成上衣", 4, 10)
        self.assertIsNone(agent.sessions.get("s").max_price)

    def test_budget_ceiling_rejects_nonfinite_negative_and_boolean_values(self):
        from shopping_copilot.react import SCHEMAS, _validate
        for value in (-1, float('nan'), float('inf'), True, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate({"max_price": value}, SCHEMAS["update_search_state"])



class DeepSeekClientTest(unittest.TestCase):
    def response(self, **changes):
        payload = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "private",
            "tool_calls": [action("search_products", query="shoes")],
        }}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
        payload.update(changes)
        return json.dumps(payload).encode()

    def test_wire_format_and_usage(self):
        handle = io.BytesIO(self.response())
        with patch("shopping_copilot.llm.urlopen", return_value=handle) as send:
            config = LLMConfig(api_key="test-secret", base_url="https://api.deepseek.com/v1")
            reply = DeepSeekClient(config).complete([{"role": "user", "content": "shoes"}], [])
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/v1/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(reply.prompt_tokens, 12)
        self.assertNotIn("reasoning_content", reply.message)
        self.assertNotIn("test-secret", repr(config))

    def test_rate_limit_retries_and_auth_failure_does_not_leak_body(self):
        limited = HTTPError("url", 429, "limited", {}, io.BytesIO(b"secret-body"))
        with patch("shopping_copilot.llm.urlopen", side_effect=[limited, io.BytesIO(self.response())]) as send, \
                patch("shopping_copilot.llm.time.sleep"):
            DeepSeekClient(LLMConfig(api_key="test")).complete([], [])
        self.assertEqual(send.call_count, 2)
        denied = HTTPError("url", 401, "secret-key", {}, io.BytesIO(b"secret-body"))
        with patch("shopping_copilot.llm.urlopen", side_effect=denied) as send:
            with self.assertRaisesRegex(LLMError, "^LLM HTTP 401$"):
                DeepSeekClient(LLMConfig(api_key="test")).complete([], [])
        self.assertEqual(send.call_count, 1)

    def test_transport_failure_and_malformed_responses(self):
        with patch("shopping_copilot.llm.urlopen", side_effect=URLError("secret")):
            with self.assertRaisesRegex(LLMError, "failed or timed out"):
                DeepSeekClient(LLMConfig(api_key="test", max_retries=0)).complete([], [])
        for raw in (b"not-json", b"{}", b"[]", self.response(choices=[]), self.response(usage={"prompt_tokens": -1})):
            with self.subTest(raw=raw), self.assertRaises(LLMError):
                DeepSeekClient._parse(raw)
        payload = json.loads(self.response())
        payload["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(LLMError):
            DeepSeekClient._parse(json.dumps(payload).encode())

    def test_configuration_bounds(self):
        for options in ({"base_url": "http://example.com"}, {"max_steps": 0}, {"timeout": float("nan")},
                        {"max_tokens": 0}, {"max_retries": 4}, {"base_url": "https://user:secret@example.com"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LLMConfig(**options)


if __name__ == "__main__":
    unittest.main()
