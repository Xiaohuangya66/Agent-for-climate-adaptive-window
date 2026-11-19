# Agent for climate-adaptive window design

Source code for an LLM-controlled agent that designs multilayer windows under building-energy and daylight requirements. A single LLM coordinates numerical tools for structure generation, optical calculation, performance prediction, building simulation and adaptation to a new city.

## Engineering workflow

1. Resolve the requested city and check the required weather, building and reference assets.
2. For a new city, prepare its clear-glass baseline and use a small set of probe simulations to adapt source-city performance models.
3. Select a spectral-target action (`keep`, `refine` or `switch`), a prediction-risk profile and an information-query allocation.
4. Generate structures with a Transformer and test-time reinforcement learning (TTT-RL), calculate spectra with the transfer-matrix method (TMM), and screen candidates with XGBoost ensembles.
5. Evaluate selected candidates with EnergyPlus, record their energy use and useful daylight illuminance (UDI), and update the search state.
6. Return the lowest-energy verified design satisfying the prescribed daylight constraint within the query budget.

The LLM selects strategies through validated tool interfaces. Numerical code enforces the simulation budget, observation roles, daylight threshold and final selection rule.

## Code organization

| Directory | Responsibility |
| --- | --- |
| `code/llm_controller/` | LLM transport, system prompt, tool schemas, state transitions, city resolution and numerical-tool coordination |
| `code/runtime/src/agent/` | Numerical search loop, candidate records, formal evaluation protocol and optimization methods |
| `code/runtime/src/components/` | TTT policy, TMM interface, surrogate interface, IDF construction and EnergyPlus execution |
| `code/runtime/src/tools/` | City support, probe planning, tree-model continuation and acceptance, target refinement, residual correction and information selection |
| `code/runtime/src/vendor/thermo_agent/` | Supporting optical, simulation, structure and policy routines |
| `code/transformer/` | Transformer architecture used by the generator |
| `code/surrogate_training/` | Grouped source-data preparation, ensemble training, calibration and model acceptance |
| `code/data_generation/` | Source-label generation and clear-glass baseline simulation utilities |
| `code/ideal_reference/` | Idealized optical-state construction and simulation utilities |
| `code/runtime/experiments/` | Numerical-method execution and engineering validation utilities |
| `inputs/reference_assets/TMM/tmm_core.py` | Supporting coherent/incoherent TMM implementation |
| `tools/` | Command-line entry points |

The active LLM workflow enters through `tools/run_llm_agent.py`. Its numerical bridge is `code/llm_controller/numerical.py`, which uses `AgentFull`. The active generator is `code/runtime/src/components/ttt_policy.py`, loading the architecture from `code/transformer/model.py`.

The NSGA-II implementation is `code/runtime/src/agent/nsga2_method.py` and uses direct EnergyPlus evaluations. The numerical `agent_full` entry is also available for engineering work; it does not invoke the LLM controller.

## Entry points

| Command-line script | Task |
| --- | --- |
| `tools/run_llm_agent.py` | Run the LLM-controlled design agent for a city or a natural-language city request |
| `tools/run_experiment.py` | Run a numerical optimization method, including the NSGA-II baseline |
| `tools/train_sources.py` | Prepare grouped partitions and train/accept source-city tree ensembles |
| `tools/calibrate_transfer.py` | Select transfer settings through source-city leave-one-city-out evaluation |
| `tools/publish_protocol.py` | Build source anchors and spectral-prior assets from prepared source data |
| `tools/prepare_assets.py` | Prepare the project asset bindings |
| `tools/run_ideal_scan.py` | Construct and optionally simulate idealized reference spectra |
| `tools/verify_llm_run.py` | Check a saved agent run against the engineering protocol |

## Environment and external assets

Use a Python environment compatible with `requirements.txt`. Install `requirements.llm.txt` when using the LLM interface. Building evaluations require a separately installed EnergyPlus 23.1.0 executable.

This repository distributes source code only. Trained weights, model configuration files, source datasets, weather/building inputs, optical data tables and simulation outputs are configured separately. The command-line execution and training tools require these external files in their expected locations.

Runtime bindings are read by `code/runtime/src/support_settings.py`. Set `CITY_SUPPORT_SETTINGS` to the local runtime-settings file, or provide `configs/runtime.json` in a local working copy. The settings bind the model architecture, checkpoints, source models, baseline table, weather/building assets and optical reference assets. `ENERGYPLUS_EXE` can select the installed executable.

The LLM adapter reads `DMX_API_KEY`, `DMX_BASE_URL` and `DMX_MODEL` from the process environment. Credentials and provider configuration values are not stored in this repository.

## Command examples

After installing dependencies, inspect the entry-point options:

```bash
python tools/run_llm_agent.py --help
python tools/run_experiment.py --help
```

After configuring the external runtime assets, a source-city agent and an NSGA-II run can be started with:

```bash
python tools/run_llm_agent.py --city Singapore --seed 5 --budget 20 --candidate-budget 384 --generation-batch 64 --output-dir results/agent_singapore_seed5
python tools/run_experiment.py --method nsga2 --city Singapore --seed 5 --budget 40 --output-dir results/nsga_singapore_seed5
```

For a target city requiring model adaptation, the LLM entry supports a 40-query workflow with initial support probes and subsequent online design:

```bash
python tools/run_llm_agent.py --city Shenzhen --seed 5 --budget 40 --candidate-budget 384 --generation-batch 128 --output-dir results/agent_shenzhen_seed5
```

Use a fresh output directory for each run. The tools maintain query records and reject accidental reuse of existing run directories.
