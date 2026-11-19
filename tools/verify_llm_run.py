"""Audit a completed local LLM run, real evidence and historical context."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def facts(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    output = args.output_dir.resolve()
    config = read(output/'run_config.json')
    run = Path(config['run_directory'])
    result = read(run/'summary.json')
    rows = facts(run/'queries.jsonl')
    dispatch = facts(run/'query_dispatch.jsonl')
    trace = facts(run/'llm_trace.jsonl')
    usage = read(run/'llm_usage.json')
    api = [e for e in trace if e['event'] == 'api_response']
    calls = [e for e in trace if e['event'] == 'tool_result']
    states_before = {c['id']: e['state_before'] for e in api for c in e.get('tool_calls', [])}
    approved = [e for e in calls if e['payload']['ok']]
    rejected = [e for e in calls if not e['payload']['ok']]
    invalid_response_turns = [i for i, e in enumerate(api) if len(e.get('tool_calls', [])) != 1]
    is_transfer = config.get('support_mode') == 'transfer' or bool(result.get('method_specific_data', {}).get('city_support'))
    gate_path = run/'city_support/tree_candidate_v002/validation_gate.json'
    gate = read(gate_path) if is_transfer and gate_path.is_file() else None
    threshold = result['protocol']['formal_direct_udi_threshold']
    feasible = [r for r in rows if r.get('energyplus_verified') and r['udi'] >= threshold]
    selected = min(feasible, key=lambda r: (r['energy'], -r['udi'])) if feasible else None
    baseline = result['protocol']['energy_baseline_J']
    plan = read(run/'city_support/probe_plan.json') if is_transfer else {'samples': []}
    evidence = []
    for row in rows:
        directory = Path(row['energyplus_result']['run_directory'])
        end = directory/'eplusout.end'
        csv_file = directory/'eplusout.csv'
        energy = row['energyplus_result']
        evidence.append({'query_index': row['query_index'], 'end_file': end.is_file(),
            'completed_successfully': end.is_file() and 'completed successfully' in end.read_text(errors='replace').lower(),
            'hourly_csv': csv_file.is_file(), 'verified': row['energyplus_verified'],
            'meter_balance': bool(row['energyplus_verified'] and math.isclose(row['energy'],
                energy['ElectricityEnergy']+energy['NaturalGasEnergy'], rel_tol=1e-10))})
    validation_ids = {r['query_index'] for r in rows if r.get('adaptation_role') == 'validation'}
    target_sources = {s for h in result['target_history'] for s in h.get('source_structures', [])}
    validation_structures = {r['structure'] for r in rows if r.get('adaptation_role') == 'validation'}
    checks = {
        'controller_completed': usage['completed'] and not result['controller']['interrupted'],
        'budget_exact': len(rows) == len(dispatch) == config['budget'] == result['n_energyplus_queries'],
        'contiguous_queries': [r['query_index'] for r in rows] == list(range(1, config['budget']+1)),
        'no_duplicate_structures': len({r['structure'] for r in rows}) == len(rows),
        'all_real_success': all(e['verified'] and e['completed_successfully'] and e['hourly_csv'] for e in evidence),
        'all_energy_meter_balance': all(e['meter_balance'] for e in evidence),
        'real_udi90': result['protocol']['udi_ratio_required'] == .9 and math.isclose(threshold,.9*result['protocol']['clear_glass_udi']),
        'formal_selected_best': (selected['structure'] == result['best_structure'] and
            math.isclose(result['best_energy'], selected['energy'], rel_tol=1e-12)) if selected else
            (not result['best_is_feasible'] and result['best_energy_saving'] is None),
        'no_rule_fallback': usage['rule_policy_fallbacks'] == 0,
        'approved_versions_contiguous': [e['payload']['state']['state_version'] for e in approved] ==
                                       list(range(1, len(approved)+1)),
        'rejected_calls_never_spend_or_advance': all(
            e['payload']['state']['state_version'] == states_before[e['tool_call_id']]['state_version'] and
            e['payload']['state']['budget_used'] == states_before[e['tool_call_id']]['budget_used'] for e in rejected),
        'rejection_count_matches_usage': len(rejected) + len(invalid_response_turns) == usage['rejected_calls'],
        'invalid_responses_never_spend_or_advance': all(
            i+1 < len(api) and all(api[i]['state_before'][key] == api[i+1]['state_before'][key]
                                  for key in ['state_version', 'budget_used'])
            for i in invalid_response_turns),
        'candidate_budget_exact': result['method_specific_data']['n_candidates_generated'] == config['candidate_budget'],
        'validation_not_target_source': not target_sources.intersection(validation_structures),
        'information_quota_exact': result['method_specific_data']['n_information_queries'] ==
                                   result['method_specific_data']['information_query_quota'],
        'hard_constraints_unchanged': result['protocol']['search_space']['required_wvo2_count'] == 0 and
                                     result['protocol']['search_space']['terminal_layer'] is None and
                                     result['protocol']['search_space']['maximum_wvo2_count'] == 2 and
                                     all(sum(t.startswith('WVO2_') for t in r['structure'].split())<=2 for r in rows),
    }
    if is_transfer:
        decision = result['method_specific_data']['city_support']['adapter_decision']['decision']
        fit_ids = set(gate['fit_query_ids']) if gate else set()
        gate_honored = (decision == ('publish_continued_trees' if gate['accepted'] else 'keep_gate_rejected')
                        if gate else decision in {'keep_insufficient_target_labels', 'keep_base_experts'})
        checks.update(probe_roles_12_4=(sum(s['adaptation_role'] == 'fit' for s in plan['samples']) == 12 and
            sum(s['adaptation_role'] == 'validation' for s in plan['samples']) == 4),
            validation_not_tree_fit=not fit_ids.intersection(validation_ids),
            tree_gate_honored=gate_honored)
    # Only the current process's supplied key is examined. Never print its value.
    secret = os.environ.get('DMX_API_KEY')
    checked = 0
    leaked = 0
    if secret:
        targets = list(output.rglob('*.json')) + list(output.rglob('*.jsonl')) + list(output.rglob('*.log'))
        targets += list((ROOT/'code/llm_controller').glob('*.py')) + [ROOT/'tools/run_llm_agent.py']
        targets += [ROOT/'docs/LLM_SINGLE_AGENT_IMPLEMENTATION.md']
        for file in targets:
            if not file.is_file():continue
            checked += 1
            leaked += int(secret in file.read_text(encoding='utf-8', errors='replace'))
        checks['credential_not_in_artifacts'] = leaked == 0
    historical = []
    catalog=ROOT/'provenance/run_catalog.json'
    for item in read(catalog) if catalog.is_file() else []:
        if item['city'] == config['city'] and item['seed'] == config['seed']:
            old = read(ROOT/item['summary'])
            historical.append({'method': item['method'], 'seed': item['seed'], 'budget': item['budget'],
                'best_saving_fraction': old['best_energy_saving'], 'best_udi': old['best_udi'],
                'n_feasible': old['n_feasible'], 'path': item['summary']})
    report = {'status': 'PASS' if all(checks.values()) else 'CHECK_REQUIRED', 'checks': checks,
        'run_directory': str(run), 'n_real_queries': len(rows), 'n_feasible': len(feasible),
        'best_saving_fraction': result['best_energy_saving'], 'best_udi': result['best_udi'],
        'best_udi_retention': result['best_udi']/result['protocol']['clear_glass_udi'] if result.get('best_udi') is not None else None,
        'best_structure': result['best_structure'], 'best_optical_features': selected['optical_properties'] if selected else None,
        'llm': usage, 'llm_rejection_audit': {'rejected_tool_calls': len(rejected),
            'invalid_response_turns': len(invalid_response_turns)},
        'returned_models': sorted({e['metadata'].get('returned_model', '') for e in api}),
        'rounds': result['controller']['rounds'], 'model_gate': gate, 'historical_context': historical,
        'historical_comparison_caveat': 'LLM run additionally excludes publication validation labels from target/acquisition/LLM feedback; historical trajectories are not an isolated LLM ablation.',
        'evidence': evidence, 'credential_scan': {'checked_files': checked, 'files_containing_key': leaked} if secret else 'not_checked_no_key_in_process'}
    (output/'validation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output/'rounds.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        names = ['round', 'target_action', 'cluster_id', 'margin_profile', 'information_requested', 'n_information',
                 'query_start', 'query_end', 'success', 'n_fit_feasible', 'best_fit_saving', 'best_cumulative_saving']
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        for entry in report['rounds']:
            before = [r for r in feasible if r['query_index'] <= entry['query_end']]
            writer.writerow({**{key: entry.get(key) for key in names if key not in
                ['target_action', 'cluster_id', 'margin_profile', 'best_cumulative_saving']},
                **{key: entry['strategy'][key] for key in ['target_action', 'cluster_id', 'margin_profile']},
                'best_cumulative_saving': max((r['energy_saving'] for r in before), default=None)})
    md = ['# 本地单 LLM 城市测试', '',
        f'城市：{config["city"]}；seed：{config["seed"]}；验收：{report["status"]}。', '',
        'API 服务配置和模型选择由进程环境提供。本报告不记录令牌或具体模型标识。', '',
        '| 指标 | 数值 |', '|---|---:|',
        f'| 真实年度查询 | {len(rows)} |', f'| 成功查询 | {sum(r["energyplus_verified"] for r in rows)} |',
        f'| UDI90 可行查询 | {len(feasible)} |', f'| 候选生成预算 | {config["candidate_budget"]} |',
        f'| API 请求数（本优化流程） | {usage["api_requests"]} |',
        f'| token（本优化流程） | {usage["token_usage"]["total_tokens"]} |',
        f'| 被拒绝并阻止执行的调用 | {usage["rejected_calls"]} |',
        f'| 规则策略降级次数 | {usage["rule_policy_fallbacks"]} |', '',
        '| 轮次 | 目标动作 | 簇 | 风险档位 | 性能 / 信息查询 | 轮末累计最高可行节能率 |',
        '|---:|---|---:|---|---:|---:|']
    for entry in report['rounds']:
        best = max((r['energy_saving'] for r in feasible if r['query_index'] <= entry['query_end']), default=None)
        strategy = entry['strategy']
        md.append(f'| {entry["round"]} | {strategy["target_action"]} | {strategy["cluster_id"]} | '
                  f'{strategy["margin_profile"]} | {entry["attempts"]-entry["n_information"]} / '
                  f'{entry["n_information"]} | {best:.4%} |' if best is not None else '| No feasible result |')
    md += ['', '| 查询 | 类型 | 真实节能率 | 真实 UDI | 清玻璃 UDI 保留比例 | 达标 |',
           '|---:|---|---:|---:|---:|---|']
    for row in rows:
        if row['energyplus_verified']:
            md.append(f'| {row["query_index"]} | {row.get("selection",{}).get("mode","probe")} | '
                f'{row["energy_saving"]:.4%} | {row["udi"]:.6f} | '
                f'{row["udi"]/result["protocol"]["clear_glass_udi"]:.4%} | {row["direct_feasible"]} |')
        else:
            md.append(f'| {row["query_index"]} | failed | — | — | — | False |')
    if selected:
        md += ['', '## 最优真实可行结构', '', '```text', selected['structure'], '```', '',
            '| 指标 | 数值 |', '|---|---:|', f'| 节能率 | {selected["energy_saving"]:.6%} |',
            f'| 能耗 kWh | {selected["energy"]/3.6e6:.6f} |',
            f'| 节省 kWh | {(baseline-selected["energy"])/3.6e6:.6f} |',
            f'| 电能 kWh | {selected["energyplus_result"]["ElectricityEnergy"]/3.6e6:.6f} |',
            f'| 天然气 kWh | {selected["energyplus_result"]["NaturalGasEnergy"]/3.6e6:.6f} |', '',
            '| 光学特征 | cold | hot |', '|---|---:|---:|']
        for metric in ['T_lum', 'T_sol', 'Rf_sol', 'Rb_sol']:
            props = selected['optical_properties']
            md.append(f'| {metric} | {props[metric+"_cold"]:.8f} | {props[metric+"_hot"]:.8f} |')
    if historical:
        md += ['', '## 已归档的同 seed 参考', '',
            '以下预算和代码快照与本次不同，属于背景对照，不是隔离 LLM 贡献的配对实验。', '',
            '| 方法 | seed | 查询预算 | 最高真实可行节能率 | 最优 UDI |', '|---|---:|---:|---:|---:|']
        for old in historical:
            md.append(f'| {old["method"]} | {old["seed"]} | {old["budget"]} | '
                      f'{old["best_saving_fraction"]:.4%} | {old["best_udi"]:.6f} |')
    md += ['', '## 审计边界', '',
        '真实结果只来自 EnergyPlus；最终选择只来自 FormalProtocol。统计不使用平均节能率。', '',
        '实际执行代码见本目录 code_snapshot.zip；运行中没有热替换。当前入口已补充归一化能耗残差与更明确的参数 Schema，单独 API 回放不产生新的仿真结果。', '',
        '理由是可审计的动作说明，不等于正确推理或性能提升的独立证明。后续需在同一数值版本、预算、候选总额和数据隔离条件下进行多 seed 配对比较。', '',
        '详细状态、门控与逐查询证据见 validation_report.json 和 method_runs 下的 llm_trace.jsonl、queries.jsonl。']
    (output/'report.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'checks', 'n_real_queries', 'n_feasible',
                                                 'best_saving_fraction', 'best_udi', 'llm']}, ensure_ascii=False, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
