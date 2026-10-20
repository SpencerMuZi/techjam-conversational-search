from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shopping_copilot.agent import ShoppingCopilotAgent
from shopping_copilot.config import AgentConfig
from shopping_copilot.llm import LLMConfig, LLMError
from shopping_copilot.models import SearchContext
from tests.test_agent_framework import PRODUCTS
from tests.test_react_agent import ScriptedClient, action


class OptimizedWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = Path(self.directory.name) / 'catalog.jsonl'
        self.write_catalog(PRODUCTS)

    def tearDown(self):
        self.directory.cleanup()

    def write_catalog(self, products):
        self.catalog.write_text(''.join(json.dumps(row) + '\n' for row in products))

    def agent(self, client, mode='react', adaptive=True, **options):
        agent = ShoppingCopilotAgent(self.catalog, AgentConfig(clarification_candidate_cutoff=0),
            mode=mode, llm_client=client,
            llm_config=LLMConfig(adaptive_workflow=adaptive, fallback=False, **options))
        agent.reset('s', {})
        return agent

    def test_finish_combines_ranked_candidates_and_exact_structured_question(self):
        client = ScriptedClient(
            [action('search_products', query='red leather shoes')],
            [action('finish', message='Catalog evidence is available.', parent_asins=['BLACK_SHOE'],
                    ask_attribute='color', question='Which color do you prefer?')],
        )
        agent = self.agent(client, adaptive=False)
        response = agent.respond('s', 'red leather shoes', 1, 10)
        self.assertEqual(response['ask_attribute'], 'color')
        self.assertTrue(response['message'].endswith('Do you have a color preference?'))
        self.assertEqual(len(response['recommendations']), 2)
        self.assertEqual([row['score'] for row in response['recommendations']],
                         sorted((row['score'] for row in response['recommendations']), reverse=True))

    def test_question_in_body_is_rejected_then_repaired(self):
        client = ScriptedClient(
            [action('finish', message='Which size?', parent_asins=[])],
            [action('finish', message='Let us narrow the candidates.', parent_asins=[],
                    ask_attribute='size', question='Which size?')],
        )
        agent = self.agent(client, adaptive=False)
        response = agent.respond('s', 'shoes', 1, 10)
        self.assertIn('error', agent.last_trace[0]['observation'])
        self.assertEqual(response['ask_attribute'], 'size')

    def test_attribute_controls_question_semantics_and_language(self):
        client = ScriptedClient([action('finish', message='需要再确认。', parent_asins=[],
                                       ask_attribute='other', question='Which sleeve length?')])
        agent = self.agent(client, adaptive=False)
        response = agent.respond('s', '帮我找上衣', 1, 10)
        self.assertEqual(response['ask_attribute'], 'other')
        self.assertTrue(response['message'].endswith('还有哪些要求或偏好需要优先考虑？'))
        self.assertNotIn('sleeve', response['message'])

    def test_logistic_default_retains_ten_when_model_proposes_two(self):
        self.write_catalog([{**PRODUCTS[0], 'parent_asin': f'SHOE_{index}'} for index in range(13)])
        client = ScriptedClient([action('search_products', query='red leather shoes')],
                                [action('finish', message='Ranked candidates.', parent_asins=['SHOE_0', 'SHOE_1'])])
        agent = self.agent(client, adaptive=False)
        response = agent.respond('s', 'red leather shoes', 1, 10)
        self.assertEqual(len(response['recommendations']), 10)

    def test_filter_experiment_is_opt_in_and_still_retains_logistic_order(self):
        client = ScriptedClient([action('search_products', query='shoes cotton')],
            [action('finish', message='Filtered candidates.', parent_asins=['BLACK_SHOE', 'LEATHER_SHOE'])])
        agent = self.agent(client, adaptive=False, ranking_policy='llm_filter')
        response = agent.respond('s', 'shoes cotton', 1, 10)
        self.assertEqual({row['parent_asin'] for row in response['recommendations']}, {'LEATHER_SHOE', 'BLACK_SHOE'})
        scores = [row['score'] for row in response['recommendations']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_expansion_keeps_original_keyword_query_and_reranks_union(self):
        agent = self.agent(ScriptedClient(), adaptive=False)
        agent.retriever.config = AgentConfig(retrieval_depth=1, rerank_depth=1, precise_seed=1)
        agent.retriever.capture = True
        context = SearchContext('buying', 'Shoes', {'material': ('red leather',)}, {}, (),
                                'original phrase red leather', expansion_queries=('cotton shirt',))
        with patch.object(agent.retriever.index, 'search_fts_scored', wraps=agent.retriever.index.search_fts_scored) as search:
            agent.retriever.retrieve(context, 10)
        queries = [call.args[0] for call in search.call_args_list]
        self.assertTrue(any('original phrase red leather' in query for query in queries))
        self.assertIn('cotton shirt', queries)
        candidates = {asin for asin, _ in agent.retriever.last_candidates}
        self.assertIn('LEATHER_SHOE', candidates)
        self.assertIn('COTTON_SHIRT', candidates)

    def test_query_expansion_requires_explicit_opt_in(self):
        self.assertFalse(LLMConfig().expansion_enabled)
        client = ScriptedClient([action('update_search_state', category='Shoes', query_expansions=['cotton shirt'])])
        agent = self.agent(client)
        with patch.object(agent.retriever.index, 'search_fts_scored', wraps=agent.retriever.index.search_fts_scored) as search:
            agent.respond('s', "I'm looking for Shoes, but I'm still exploring.", 1, 10)
        self.assertNotIn('cotton shirt', [call.args[0] for call in search.call_args_list])

    def test_explicit_react_search_still_adds_recall_with_auto_expansion_off(self):
        client = ScriptedClient([action('search_products', query='cotton shirt')],
                                [action('finish', message='Catalog candidates.', parent_asins=['COTTON_SHIRT'])])
        agent = self.agent(client, adaptive=False)
        with patch.object(agent.retriever.index, 'search_fts_scored', wraps=agent.retriever.index.search_fts_scored) as search:
            agent.respond('s', '找一件上衣', 1, 10)
        self.assertIn('cotton shirt', [call.args[0] for call in search.call_args_list])

    def test_informative_request_uses_one_parse_and_existing_question_policy(self):
        client = ScriptedClient([action('update_search_state', route='buying', category='sneakers',
                                       hard_slots={'material': ['leather']}, query_expansions=['leather sneakers'])])
        agent = self.agent(client)
        message = "I'm looking for Shoes Fashion Sneakers. A key requirement is: genuine leather."
        response = agent.respond('s', message, 1, 10)
        state = agent.sessions.get('s')
        self.assertEqual(state.category, 'Shoes Fashion Sneakers')
        self.assertEqual(state.hard_slots['material'], ['genuine leather'])
        self.assertEqual(len(client.messages), 1)
        self.assertEqual(response['ask_attribute'], 'other')
        self.assertIn('What other requirements', response['message'])
        self.assertEqual(len(response['recommendations']), 2)
        self.assertEqual(len(state.history), 2)

    def test_raw_same_attribute_phrases_accumulate_across_turns(self):
        client = ScriptedClient(
            [action('update_search_state', category='Shoes', route='buying', hard_slots={'material': ['leather']})],
            [action('update_search_state', hard_slots={'material': ['soft lining']})],
        )
        agent = self.agent(client, mode='llm_parse')
        agent.respond('s', "I'm looking for Shoes. A key requirement is: genuine leather.", 1, 10)
        agent.respond('s', 'For that, what matters is: soft leather lining.', 2, 10)
        self.assertEqual(agent.sessions.get('s').hard_slots['material'], ['genuine leather', 'soft leather lining'])

    def test_empty_inverse_bucket_is_a_valid_move_and_override_drops_old_intent(self):
        client = ScriptedClient(
            [action('update_search_state', category='Shoes', soft_slots={'feature': ['Hand Wash Only']})],
            [action('update_search_state', hard_slots={'material': ['polyester']},
                    soft_slots={'material': [], 'feature': ['Hand Wash Only']})],
        )
        agent = self.agent(client, mode='llm_parse')
        agent.respond('s', "I'm looking for Shoes. Hand Wash Only.", 1, 10)
        agent.respond('s', 'Actually, ignore my earlier preference. What I need is: polyester.', 2, 10)
        state = agent.sessions.get('s')
        self.assertEqual(state.hard_slots, {'material': ['polyester']})
        self.assertFalse(state.soft_slots)

    def test_conflicting_nonempty_hard_and_soft_values_are_still_invalid(self):
        from shopping_copilot.react import normalize_slot_updates
        with self.assertRaises(ValueError):
            normalize_slot_updates({'hard_slots': {'color': ['red']}, 'soft_slots': {'color': ['blue']}})

    def test_llm_parse_mode_disables_expansion_route(self):
        client = ScriptedClient([action('update_search_state', category='Shoes', query_expansions=['cotton shirt'])])
        agent = self.agent(client, mode='llm_parse')
        with patch.object(agent.retriever.index, 'search_fts_scored', wraps=agent.retriever.index.search_fts_scored) as search:
            agent.respond('s', "I'm looking for Shoes, but I'm still exploring.", 1, 10)
        self.assertNotIn('cotton shirt', [call.args[0] for call in search.call_args_list])

    def test_insufficient_candidates_trigger_real_react_after_initial_parse(self):
        self.write_catalog([])
        client = ScriptedClient([action('update_search_state', category='Shoes')],
                                [action('ask_user', attribute='feature', question='Which feature matters most?')])
        agent = self.agent(client)
        response = agent.respond('s', 'Help me buy shoes', 1, 10)
        self.assertEqual(len(client.messages), 2)
        self.assertEqual(response['ask_attribute'], 'feature')
        self.assertTrue(any(event['tool'] == 'exploration_gate' for event in agent.last_trace))
        self.assertEqual(len(agent.sessions.get('s').history), 2)

    def test_parse_failure_rolls_back_and_reports_usage(self):
        client = ScriptedClient([action('finish', message='bad', parent_asins=[])],
                                [action('finish', message='bad', parent_asins=[])])
        agent = self.agent(client)
        with self.assertRaisesRegex(LLMError, 'extraction failed'):
            agent.respond('s', 'shoes', 1, 10)
        self.assertEqual(agent.controller.usage['prompt_tokens'], 200)
        self.assertFalse(agent.sessions.get('s').history)

    def test_approximate_budget_is_not_reinterpreted_as_hard_ceiling(self):
        client = ScriptedClient([action('update_search_state', route='buying', category='Shoes',
                                       hard_slots={'budget': ['under 40']}, max_price=40)])
        agent = self.agent(client, mode='llm_parse')
        response = agent.respond('s', "I'm looking for Shoes. A key requirement is: budget around $40.", 1, 10)
        self.assertIsNone(agent.sessions.get('s').max_price)
        self.assertTrue(response['recommendations'])

    def test_budget_filter_backfills_and_keeps_rank_order(self):
        self.write_catalog([{**PRODUCTS[0], 'parent_asin': f'SHOE_{index}', 'price': 100 if index < 5 else 30}
                            for index in range(18)])
        client = ScriptedClient([action('update_search_state', route='buying', category='Shoes',
                                       hard_slots={'budget': ['under 50']}, max_price=50)])
        agent = self.agent(client, mode='llm_parse')
        response = agent.respond('s', 'Shoes under 50', 1, 10)
        self.assertEqual(len(response['recommendations']), 10)
        self.assertTrue(all(agent.retriever.index.documents[row['parent_asin']].price <= 50 for row in response['recommendations']))


if __name__ == '__main__':
    unittest.main()
