"""
单次实验运行脚本

用法:
    python experiments/run_single.py --method agent_full --city Singapore --seed 0 --budget 40
"""

import sys
import argparse
import json

import numpy as np

def convert_to_json_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

import time
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.evaluation import summarize_results


AVAILABLE_METHODS = [
    'agent_full',           # 完整方法（城市先验+在线修正）
    'city_prior_no_refine', # 仅城市先验，无在线修正
    'fixed_target',         # 固定目标（AM1.5）
    'nsga2',               # NSGA-II多目标优化
    'ttt_rl',              # TTT-RL
    'random_search',       # 随机搜索基线
]

from src.runtime_profile import registry
AVAILABLE_CITIES = list(registry()['source_cities']) + [c for c,v in registry()['cities'].items() if v.get('role')=='target' and 'idf' in v]


def parse_args():
    parser = argparse.ArgumentParser(description='运行单次实验')

    parser.add_argument('--method', type=str, required=True,
                       choices=AVAILABLE_METHODS,
                       help='优化方法')

    parser.add_argument('--city', type=str, required=True,
                       choices=AVAILABLE_CITIES,
                       help='目标城市')

    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子')

    parser.add_argument('--budget', type=int, default=40,
                       help='EnergyPlus查询预算')

    parser.add_argument('--udi-ratio', type=float, default=.90,
                       help='Required direct UDI / same-city clear-glass UDI, e.g. 0.90')

    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')

    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')

    parser.add_argument('--population-size', type=int, default=20)
    parser.add_argument('--candidate-budget', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-nhead', type=int, choices=[4,8], default=4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-layers', type=int, default=None)
    parser.add_argument('--fitness-source', choices=['surrogate','direct'], default='direct')
    parser.add_argument('--agent-batch-size', type=int, default=128)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Explicit initial checkpoint for Agent or TTT; otherwise use method default')
    parser.add_argument('--agent-selection', choices=['hybrid', 'performance'], default='hybrid',
                       help='Budgeted information-value queries, or the previous performance-only selector')
    parser.add_argument('--information-fraction', type=float, default=.3,
                       help='Fraction of total design budget; quota is spread over online queries after probes, first online query stays performance')
    parser.add_argument('--probe-count', type=int, default=None,
                       help='Transfer probes included within the same design-query budget')
    return parser.parse_args()


def load_agent(method: str, city: str, seed: int, budget: int, options=None):
    """根据方法名称加载对应的Agent"""

    from src.agent.formal_protocol import configure_udi_ratio
    configure_udi_ratio(getattr(options,'udi_ratio',.90))

    if method == 'agent_full':
        from src.agent.agent_full import AgentFull
        return AgentFull(city=city, seed=seed, budget=budget,
            candidate_budget=getattr(options,'candidate_budget',384),
            batch_size=getattr(options,'agent_batch_size',128),
            ttt_batch_size=getattr(options,'batch_size',16),
            learning_rate=getattr(options,'learning_rate',1e-4),
            temperature=getattr(options,'temperature',1.), nhead=getattr(options,'model_nhead',4),
            checkpoint_path=getattr(options,'checkpoint',None),
            selection_mode=getattr(options,'agent_selection','hybrid'),
            information_fraction=getattr(options,'information_fraction',.3),
            probe_count=getattr(options,'probe_count',None),
            max_layers=getattr(options,'max_layers',None),
            output_root=Path(options.output_dir)/'method_runs' if options else None)

    elif method == 'city_prior_no_refine':
        from src.agent.city_prior_no_refine import CityPriorNoRefine
        return CityPriorNoRefine(city=city, seed=seed, budget=budget)

    elif method == 'fixed_target':
        from src.agent.fixed_target import FixedTarget
        return FixedTarget(city=city, seed=seed, budget=budget)

    elif method == 'nsga2':
        from src.agent.nsga2_method import NSGA2Method
        return NSGA2Method(city=city, seed=seed, budget=budget, population_size=getattr(options, 'population_size', 20), candidate_budget=getattr(options,'candidate_budget',384), fitness_source=getattr(options,'fitness_source','direct'), max_layers=getattr(options, 'max_layers', None), probe_count=getattr(options,'probe_count',None), output_root=Path(options.output_dir) / 'method_runs' if options else None)

    elif method == 'ttt_rl':
        from src.agent.ttt_rl_method import TTTRLMethod
        return TTTRLMethod(city=city, seed=seed, budget=budget, candidate_budget=getattr(options, 'candidate_budget', 384), batch_size=getattr(options, 'batch_size', 16), nhead=getattr(options, 'model_nhead', 4), temperature=getattr(options, 'temperature', 1.0), learning_rate=getattr(options, 'learning_rate', 1e-4), checkpoint_path=getattr(options,'checkpoint',None), max_layers=getattr(options, 'max_layers', None), output_root=Path(options.output_dir) / 'method_runs' if options else None)

    elif method == 'random_search':
        from src.agent.random_search import RandomSearch
        return RandomSearch(city=city, seed=seed, budget=budget)

    else:
        raise ValueError(f"Unknown method: {method}")


def run_single_experiment(args):
    """运行单次实验"""
    
    # 配置日志输出到文件和控制台
    import logging
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"run_{args.method}_{args.city}_seed{args.seed}_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")

    print("="*60)
    print(f"实验配置")
    print("="*60)
    print(f"方法: {args.method}")
    print(f"城市: {args.city}")
    print(f"随机种子: {args.seed}")
    print(f"预算: {args.budget}")
    print(f"UDI标准: 清玻璃UDI的 {getattr(args,'udi_ratio',.95):.1%}")
    print("="*60)

    # 加载Agent
    print("\n加载Agent...")
    agent = load_agent(args.method, args.city, args.seed, args.budget, options=args)
    print(f"✓ Agent加载成功: {agent.__class__.__name__}")

    # 运行
    print(f"\n开始运行...")
    print(f"EnergyPlus查询预算: {args.budget}; 实际耗时由仿真与搜索决定")

    start_time = time.time()

    try:
        result = agent.run()
        elapsed_time = time.time() - start_time

        print("\n" + "="*60)
        print("运行完成！")
        print("="*60)

        # 显示结果
        print(f"用时: {elapsed_time/3600:.2f} 小时")
        if result.get('best_is_feasible') is False:
            print(f"未找到满足 Clear glass UDI {agent.formal.udi_ratio:.0%} 的可行结构。")
            print("最高UDI诊断结构:", result.get('best_udi'))
        else:
            print("最优节能率:", result.get("best_energy_saving"))
            print("最优UDI:", result.get("best_udi"))
        if 'formal_direct_udi_threshold' in result.get('protocol', {}):
            print("UDI达标阈值:", result['protocol']['formal_direct_udi_threshold'])
        print(f"EnergyPlus查询数: {result['n_energyplus_queries']}")

        if 'method_specific_data' in result:
            print(f"\n方法特定数据:")
            for key, value in result['method_specific_data'].items():
                print(f"  {key}: {value}")

        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 添加时间戳避免覆盖
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{args.method}_{args.city}_seed{args.seed}_{timestamp}.json"

        result['config'] = {
            'method': args.method,
            'city': args.city,
            'seed': args.seed,
            'budget': args.budget,
            'udi_ratio': agent.formal.udi_ratio,
            'elapsed_time': elapsed_time,
            'candidate_budget': args.candidate_budget,
            'max_layers': args.max_layers,
            'fitness_source': args.fitness_source if args.method == 'nsga2' else None
        }
        if hasattr(agent,'runtime_assets'):
            result['config']['runtime_assets']=agent.runtime_assets
        if args.method == 'agent_full':
            result['config'].update(agent_selection=getattr(args, 'agent_selection', 'hybrid'),
                                    information_fraction=getattr(args, 'information_fraction', .3))

        with open(output_file, 'w') as f:
            result = convert_to_json_serializable(result)
            json.dump(result, f, indent=2, allow_nan=False)

        print(f"\n结果已保存: {output_file}")

        if result.get('status') == 'failed':
            return None
        return result

    except Exception as e:
        print(f"\n✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    args = parse_args()
    result = run_single_experiment(args)

    if result is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
