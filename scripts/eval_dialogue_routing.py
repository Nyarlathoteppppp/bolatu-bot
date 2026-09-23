"""Opt-in real API evaluation. Classifies only; never executes tools or sends QQ messages.

PYTHONPATH=. python3 scripts/eval_dialogue_routing.py --output /tmp/routing.json
Cases are annotated expectations for tool need, not scores for human-likeness.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv
from qq_social_agent.config import load_config
from qq_social_agent.deepseek_client import DeepSeekClient
from qq_social_agent.jev_client import set_jev_telemetry_recorder

ROOT = Path(__file__).resolve().parent.parent


async def evaluate(args):
    load_dotenv(args.env_file or ROOT / '.env')
    cases=json.loads((ROOT / 'tests/fixtures/dialogue_routing_cases.json').read_text())
    client=DeepSeekClient(load_config().deepseek)
    semaphore=asyncio.Semaphore(2)
    results=[]
    models=set()
    def record(event):
        if event.get('model'):
            models.add(str(event['model']))
    set_jev_telemetry_recorder(record)
    async def run(case):
        async with semaphore:
            started=time.monotonic()
            try:
                result=await client.route_tool_use(
                    persona=SimpleNamespace(name='风雪',decision_prompt='群聊自然接话，先回应当前请求'),
                    recent_messages=[],current_text=case['text'],current_nickname='测试群友',
                    addressed=True,decision_action='answer',decision_reason='offline_evaluation',
                )
                # A market classification is only usable with actual symbols.
                correct=result.tool==case['expected'] and (result.tool!='market' or bool(result.symbols))
                row={**case,'actual':result.tool,'query':result.query,
                     'symbols':[s.symbol for s in result.symbols], 'correct':correct}
            except Exception as exc:
                row={**case,'correct':False,'error':type(exc).__name__}
            row['latency_ms']=round((time.monotonic()-started)*1000)
            results.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    try:
        await asyncio.gather(*(run(case) for case in cases))
    finally:
        set_jev_telemetry_recorder(None)
        await client.jev_client.aclose()
        for api in client.clients.values():
            await api.close()
    summaries={}
    for group in ['development','holdout','replay']:
        subset=[r for r in results if r['set']==group]
        summaries[group]={'correct':sum(r['correct'] for r in subset),'total':len(subset),
                          'p50_ms':statistics.median(r['latency_ms'] for r in subset)}
    artifact={'jev_models':sorted(models),'configured_search_route':client.current_route('search').label,
              'summaries':summaries,'results':results}
    if args.output:
        Path(args.output).write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    print('SUMMARY',json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--env-file')
    parser.add_argument('--output')
    asyncio.run(evaluate(parser.parse_args()))
