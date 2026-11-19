"""Run one API-controlled experiment. Provider credentials/model come ONLY from env."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys

from bundle_env import runtime_environment

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(runtime_environment(ROOT))
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/runtime')]

from llm_controller.api import ChatClient
from llm_controller.city_assets import CityAssetService, write_json
from llm_controller.prompts import PROMPT_VERSION, SYSTEM_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('request', nargs='?', help='包含城市的自然语言请求，例如：请为杭州优化节能窗')
    p.add_argument('--request', dest='request_text', help='自然语言启动请求，与位置参数二选一')
    p.add_argument('--city', help='可选的固定城市名，无城市白名单限制')
    p.add_argument('--epw', type=Path, help='显式指定本次城市的天气文件')
    p.add_argument('--ddy', type=Path, help='显式指定同站设计日文件')
    p.add_argument('--weather-dir', type=Path, action='append', default=[], help='额外的天气检索目录，可重复')
    p.add_argument('--city-catalog', type=Path, help='城市别名、天气和建筑模板目录配置')
    p.add_argument('--runtime-settings', type=Path, help='复用先前生成的城市运行配置和清玻璃基线')
    p.add_argument('--preflight', action='store_true', help='只检查 --city 的资产，不调用 API 或 EnergyPlus')
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--budget', type=int)
    p.add_argument('--candidate-budget', type=int, default=384)
    p.add_argument('--generation-batch', type=int)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.request and args.request_text:p.error('Use either the positional request or --request, not both')
    request = args.request_text or args.request
    if not args.city and not request:p.error('Provide --city or a request containing a city')
    if args.preflight and not args.city:p.error('--preflight requires --city; natural-language extraction uses the LLM')
    if args.budget is not None and args.budget < 1:p.error('--budget must be positive')
    if args.candidate_budget < 1 or (args.generation_batch is not None and args.generation_batch < 1):
        p.error('Candidate and generation budgets must be positive')
    # Inspect/resolve the city BEFORE importing Agent/FormalProtocol/runtime_profile.
    label = 'city_request'
    service = CityAssetService(ROOT, ROOT/'new_runs', epw=args.epw, ddy=args.ddy,
        weather_dirs=args.weather_dir, catalog=args.city_catalog, runtime_settings=args.runtime_settings)
    if args.city:label = service.canonical_city(args.city)
    output = args.output_dir or ROOT/'new_runs'/f'llm_{label}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}'
    output = output.resolve()
    service.output = output
    if args.preflight:
        result = service.inspect(args.city)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_dir:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output/'city_preflight.json', result)
        return
    client = ChatClient()  # Only normal LLM execution requires provider environment.
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s',
        handlers=[logging.FileHandler(output/'run.log', encoding='utf-8'), logging.StreamHandler()])
    request = request or f'请为 {label} 寻找满足同城清玻璃 UDI90 的最高真实节能窗结构。'
    write_json(output/'run_config.json', {'request': request, 'city_hint': args.city,
        'seed': args.seed, 'phase': 'city_request', 'prompt_version': PROMPT_VERSION,
        'provider_configuration': 'environment_only'})
    (output/'system_prompt.txt').write_text(SYSTEM_PROMPT, encoding='utf-8')
    from llm_controller.city_bootstrap import resolve_city_request
    outcome, messages = resolve_city_request(service, client, request, args.city,
        run_options={'budget': args.budget, 'generation_batch': args.generation_batch,
                     'candidate_budget': args.candidate_budget})
    if outcome['status'] != 'ready':
        write_json(output/'result.json', outcome)
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return
    city_context = service.context
    city = city_context['city']
    budget, batch = outcome['design_spec']['budget'], outcome['design_spec']['batch_size']
    overlay = service.activate()
    # Imports below intentionally occur only after the new-city protocol is active.
    from src.agent.formal_protocol import configure_udi_ratio
    from llm_controller.numerical import LLMNumericalAgent
    from llm_controller.controller import run_controller
    configure_udi_ratio(.9)
    agent = LLMNumericalAgent(city, args.seed, budget=budget, candidate_budget=args.candidate_budget,
        batch_size=batch, ttt_batch_size=16, max_layers=20, probe_count=16,
        output_root=output/'method_runs', information_fraction=.3)
    agent.city_resolution = city_context
    agent._phase1_initialize()
    shutil.copyfile(outcome['trace'], agent.run_directory/'llm_trace.jsonl')
    metadata = {'city': city, 'city_role': city_context['role'], 'support_mode': city_context['support_mode'],
        'request': request, 'prompt_version': PROMPT_VERSION, 'runtime_settings': str(overlay),
        'reference_baseline_processes_this_run': city_context.get('reference_baseline_processes', 0),
        'seed': args.seed, 'budget': budget,
        'candidate_budget': args.candidate_budget, 'batch_size': batch, 'udi_ratio': .9,
        'max_layers': 20, 'hard_wvo2': False, 'append_substrate': False,
        'wvo2_count_range': [0, 2], 'report_query_checkpoints': [20, 40],
        'run_directory': str(agent.run_directory), 'entry': 'tools/run_llm_agent.py',
        'provider_configuration': 'environment_only', 'historical_rule_agent_modified': False}
    write_json(output/'run_config.json', metadata)
    print('LLM_RUN_DIRECTORY=' + str(agent.run_directory), flush=True)
    result = run_controller(agent, client, initial_messages=messages,
        initial_version=outcome['version'], prior_rejections=outcome['rejected_calls'])
    (output/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result.get(k) for k in ['status', 'best_energy_saving', 'best_udi',
          'best_is_feasible', 'n_energyplus_queries', 'n_failed_queries']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
