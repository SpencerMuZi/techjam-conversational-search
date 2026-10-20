"""Small, stratified, cached ablation using the unchanged public evaluator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as evaluator
from shopping_copilot.llm import DeepSeekClient, LLMConfig, LLMError, LLMReply
from starter.agent import Agent


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def stratified_samples(samples, count, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['scenario_type']].append(sample)
    quota = {key: len(rows) * count // len(samples) for key, rows in grouped.items()}
    remainder = sorted(grouped, key=lambda key: (-(len(grouped[key]) * count % len(samples)), key))
    for key in remainder[:count - sum(quota.values())]:
        quota[key] += 1
    selected = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: row['sample_id'])
        rng.shuffle(rows)
        selected.extend(rows[:quota[key]])
    return sorted(selected, key=lambda row: row['sample_id'])


class CachedClient:
    def __init__(self, config, cache, budget, lock, max_calls):
        self.client = DeepSeekClient(config)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget, self.lock, self.max_calls = budget, lock, max_calls
        self.stats = Counter()
        self.exhausted = False

    def complete(self, messages, tools):
        payload = {'model': self.client.config.model, 'base_url': self.client.config.base_url,
                   'max_tokens': self.client.config.max_tokens, 'messages': messages, 'tools': tools}
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.json')
        if path.exists():
            data = json.loads(path.read_text())
            self.stats['cache_hits'] += 1
            # Replayed model tokens are recorded separately from actual new use.
            self.stats['replayed_prompt_tokens'] += data['prompt_tokens']
            self.stats['replayed_completion_tokens'] += data['completion_tokens']
            return LLMReply(data['message'], 0, 0)
        with self.lock:
            if self.budget['calls'] >= self.max_calls:
                self.exhausted = True
                raise LLMError('Experiment API-call budget exhausted')
            self.budget['calls'] += 1
        self.stats['api_calls'] += 1
        reply = self.client.complete(messages, tools)
        self.stats['prompt_tokens'] += reply.prompt_tokens
        self.stats['completion_tokens'] += reply.completion_tokens
        save(path, {'message': reply.message, 'prompt_tokens': reply.prompt_tokens,
                    'completion_tokens': reply.completion_tokens})
        return reply


def initialize(catalog, output, budget, lock, max_calls):
    global AGENT, CLIENT, CATALOG, CATEGORIES, PRODUCTS, OUTPUT, CONFIG
    OUTPUT = Path(output)
    CONFIG = replace(LLMConfig.from_env(), fallback=False)
    CLIENT = CachedClient(CONFIG, OUTPUT / 'cache', budget, lock, max_calls)
    AGENT = Agent(catalog, mode='react', llm_client=CLIENT, llm_config=CONFIG)
    CATALOG, CATEGORIES, PRODUCTS = evaluator.catalog_index(catalog)


def run_sample(index, sample, filter_experiment):
    arms = ['offline', 'llm_parse', 'react', 'react_expansion']
    if filter_experiment:
        arms.append('react_llm_filter')
    records = {}
    controller = AGENT.controller
    for arm in arms:
        path = OUTPUT / f'{index:03}-{arm}.json'
        if path.exists():
            records[arm] = json.loads(path.read_text())
            continue
        if CLIENT.exhausted:
            break
        AGENT.mode = 'react' if arm.startswith('react') else arm
        AGENT.controller = None if arm == 'offline' else controller
        controller.config = replace(CONFIG,
            ranking_policy='llm_filter' if arm == 'react_llm_filter' else 'logistic',
            expansion_enabled=arm in {'react_expansion', 'react_llm_filter'})
        turns = []
        started = time.monotonic()
        before = CLIENT.stats.copy()

        class Recorder:
            def reset(self, session_id, profile):
                AGENT.reset(session_id, profile)

            def respond(self, session_id, message, turn, top_k):
                began = time.monotonic()
                row = {'turn': turn, 'user_message': message, 'retrieval_pairs': []}
                retrieve = AGENT.retriever.retrieve

                def paired_retrieval(context, count, config_override=None):
                    result = retrieve(context, count, config_override)
                    if context.expansion_queries:
                        raw = retrieve(replace(context, expansion_queries=()), count, config_override)
                        row['retrieval_pairs'].append({
                            'raw_ids': [asin for asin, _ in raw.recommendations[:10]],
                            'expanded_ids': [asin for asin, _ in result.recommendations[:10]],
                        })
                    return result

                AGENT.retriever.retrieve = paired_retrieval
                try:
                    response = AGENT.respond(session_id, message, turn, top_k)
                    row['response'] = response
                    row['status'] = 'ok'
                    return response
                except Exception as error:
                    row['status'] = 'error'
                    row['error'] = str(error) if isinstance(error, LLMError) else type(error).__name__
                    raise
                finally:
                    AGENT.retriever.retrieve = retrieve
                    row['elapsed_seconds'] = round(time.monotonic() - began, 3)
                    row['tools'] = [{key: event[key] for key in ('tool', 'phase', 'model_proposed_ids') if key in event}
                                    | ({'error': event['observation']['error']} if 'error' in event['observation'] else {})
                                    for event in AGENT.last_trace]
                    turns.append(row)

        result = evaluator.evaluate(Recorder(), [sample], CATALOG, CATEGORIES, PRODUCTS)
        record = {'index': index, 'sample_id': sample['sample_id'], 'arm': arm,
                  'complete': not CLIENT.exhausted, 'evaluation': result, 'turns': turns,
                  'elapsed_seconds': round(time.monotonic() - started, 3),
                  'actual_usage': dict(CLIENT.stats - before)}
        save(path, record)
        records[arm] = record
        AGENT.sessions._states.clear()
    return {'sample_id': sample['sample_id'], 'arms': {
        arm: {'hit': record['evaluation']['sessions'][0]['hit'],
              'score': record['evaluation']['recommended_technical_score'],
              'api_calls': record['actual_usage'].get('api_calls', 0)} for arm, record in records.items()}}


def aggregate(records):
    sessions = [record['evaluation']['sessions'][0] for record in records]
    result = evaluator.metric_summary(sessions)
    efficiency = max(0, min(1, (11 - result['mttc']) / 10))
    return {**result, 'efficiency': round(efficiency, 6),
            'recommended_technical_score': round(.5 * result['hit_rate_at_10'] + .3 * result['mrr'] + .2 * efficiency, 6),
            'actual_usage': dict(sum((Counter(record['actual_usage']) for record in records), Counter())),
            'error_turns': sum(turn['status'] == 'error' for record in records for turn in record['turns']),
            'sessions': sessions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', default=str(ROOT / 'data/catalog.jsonl'))
    parser.add_argument('--dataset', default=str(ROOT / 'data/public_set.jsonl'))
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--filter-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-api-calls', type=int, default=250)
    args = parser.parse_args()
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                os.environ[name] = value
    os.environ['SHOPPING_COPILOT_LLM_FALLBACK'] = 'false'
    os.environ['SHOPPING_COPILOT_RERANKER'] = 'logistic'
    os.environ['SHOPPING_COPILOT_DEFERRAL'] = 'adaptive'
    if not 4 <= args.samples <= 40 or not 0 <= args.filter_samples <= args.samples:
        parser.error('Use 4–40 samples and no more filter samples than the total')
    samples = stratified_samples(evaluator.load_jsonl(args.dataset), args.samples, args.seed)
    # Filter ablation includes each available scenario before adding more cases.
    filter_ids = []
    for scenario in sorted({sample['scenario_type'] for sample in samples}):
        filter_ids.append(next(sample['sample_id'] for sample in samples if sample['scenario_type'] == scenario))
    filter_ids = list(dict.fromkeys([*filter_ids, *(sample['sample_id'] for sample in samples)]))[:args.filter_samples]
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((ROOT / 'shopping_copilot').glob('*.py'))}
    fingerprint['scripts/compare_llm_modes.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = {'sample_ids': [sample['sample_id'] for sample in samples], 'filter_ids': filter_ids,
                 'seed': args.seed, 'source_sha256': fingerprint,
                 'model': LLMConfig.from_env().model,
                 'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()}
    metadata = args.output / 'metadata.json'
    if metadata.exists():
        assert json.loads(metadata.read_text()) == signature, 'Checkpoint signature differs'
    else:
        save(metadata, signature)
    print(json.dumps({'samples': len(samples), 'scenarios': dict(Counter(row['scenario_type'] for row in samples)),
                      'filter_samples': len(filter_ids), 'max_api_calls': args.max_api_calls}), flush=True)
    started = time.monotonic()
    with multiprocessing.Manager() as manager:
        budget, lock = manager.dict(calls=0), manager.Lock()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                initargs=(str(args.catalog), str(args.output), budget, lock, args.max_api_calls)) as pool:
            futures = [pool.submit(run_sample, index, sample, sample['sample_id'] in filter_ids)
                       for index, sample in enumerate(samples)]
            for count, future in enumerate(as_completed(futures), 1):
                print(json.dumps({'completed': count, 'total': len(samples), **future.result()}), flush=True)
        api_calls = budget['calls']
    grouped = defaultdict(list)
    for path in args.output.glob('[0-9][0-9][0-9]-*.json'):
        record = json.loads(path.read_text())
        if record['complete']:
            grouped[record['arm']].append(record)
    summary = {'complete': all(len(grouped[arm]) == len(samples) for arm in ('offline', 'llm_parse', 'react', 'react_expansion')),
               'arms': {arm: aggregate(records) for arm, records in grouped.items()},
               'api_calls_this_process': api_calls, 'elapsed_seconds': round(time.monotonic() - started, 2)}
    save(args.output / 'summary.json', summary)
    print(json.dumps({**summary, 'arms': {arm: {key: value for key, value in result.items() if key != 'sessions'}
                                       for arm, result in summary['arms'].items()}}, ensure_ascii=False, indent=2))
    if not summary['complete']:
        raise SystemExit('API budget exhausted; incomplete arms are not reported as full-sample scores')


if __name__ == '__main__':
    main()
