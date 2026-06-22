# LongMemEval-V2 - ThinHarness Benchmark

## Context

I used this fork to test whether ThinHarness is a strong general-purpose agent harness on a nontrivial non-coding task: competitive information retrieval over long trajectory haystacks.

I scoped the comparison to the 127 dynamic questions in the small tier: `dynamic-environment` and `dynamic-environment-abs` across the web and enterprise domains. After reviewing the benchmark categories, this subset looked like the best retrieval test for my purposes: given a long history of interaction traces, can the memory layer find the state change, UI behavior, or environment fact needed to answer the question?

## Run Details

To get a local baseline for more detailed metrics than the leaderboard provides, I ran AgentRunbook-C with the resumable wrapper in `evaluation/scripts/run_agentrunbook_c_dynamic.sh`. That script plans the 127-question dynamic set, runs one question per output directory, and can be re-run to fill only missing questions. Running one question at a time is slower but was helpful when encountering intermittent issues running on my local machine. The matching ThinHarness wrapper is `evaluation/scripts/run_thinharness_dynamic.sh`.

This is a best-faith reproduction rather than an exact reproduction of the paper. The paper setup expects a local `Qwen/Qwen3.5-9B` reader deployment; I used `qwen/qwen3.5-9b` through OpenRouter. The paper uses Codex v0.117.0 for Codex and AgentRunbook-C; my rerun used the local Codex CLI available, `codex-cli 0.141.0`. Both the AgentRunbook-C rerun and ThinHarness run used `gpt-5.4-mini` with `xhigh` reasoning for the query-time memory agent and `gpt-5.2` for the evaluator.

For the ThinHarness run, I used only its generic built-in filesystem tools (`read`, `search`, `jsonl_search`, `list`, and `glob`). I did restructure the memory files into a JSONL-friendly corpus so its generic `jsonl_search` tool could work well; that seems acceptable here because AgentRunbook-C also creates a custom trajectory structure rather than using the vanilla Codex raw layout. I tried to keep the query-time system prompt as close as practical to AgentRunbook-C, changing the tool instructions only where ThinHarness needed to know how to use its built-in tools.

## Results

Across all 127 dynamic questions in the small tier:

| Method | Dynamic score | Non-abstention | Abstention | Memory query time | Memory-agent tokens / usage |
| --- | ---: | ---: | ---: | ---: | ---: |
| AgentRunbook-C rerun | 72.4% (92/127) | 86.0% (74/86) | 43.9% (18/41) | 151.9s avg, 129.1s median | 114.77M input, 1.32M output, 116.09M total |
| ThinHarness | 74.0% (94/127) | 84.9% (73/86) | 51.2% (21/41) | 99.7s avg, 87.9s median | 60.14M input, 2.10M output, 62.24M total |


The 72.4% accuracy for AgentRunbook-C matches the paper, but I would not treat this single consolidated run as a statistically signficant claim that ThinHarness has higher accuracy than AgentRunbook-C--I saw meaningful variance on a portion of the questions when doing targeted reruns. The result does make me reasonably confident that ThinHarness at least matches AgentRunbook-C's performance on this slice, and the published leaderboard reference for vanilla Codex is materially lower than both.

The memory query time is the harness-measured time around `memory.query(...)`: it includes the query-time memory retrieval agent, but not the downstream reader, scorer, or prior runtime input generation. The timing comparison isn't perfect (local Codex CLI vs. OpenAI API), but the 46.4% lower token usage indicates that 34.4% time savings is probably in the right ballpark. Note that the paper only provides a single aggregate query time figure across all questions, 108.3s, which is far lower than the 151.9s above but includes all questions in the small tier (some of which may have been faster).


# Original README Content

<p align="center">
  <a href="https://xiaowu0162.github.io/longmemeval-v2/"><img src="https://img.shields.io/badge/🌐-Website-2a75d0?style=flat-square" height="23"></a>
  <a href="https://arxiv.org/pdf/2605.12493.pdf"><img src="https://img.shields.io/badge/📝-Paper-d03c36?style=flat-square" height="23"></a>
  <a href="https://huggingface.co/datasets/xiaowu0162/longmemeval-v2" ><img src="https://img.shields.io/badge/🤗-Data-167f5f?style=flat-square" height="23"></a>
  <a href="https://xiaowu0162.github.io/longmemeval-v2/#leaderboard" ><img src="https://img.shields.io/badge/🏆-Leaderboard-d89216?style=flat-square" height="23"></a>
</p>

**LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues**

[Di Wu](https://xiaowu0162.github.io/),
[Zixiang Ji](https://www.linkedin.com/in/zixiang-ji-56902624b/),
[Asmi Kawatkar](https://www.linkedin.com/in/asmi-kawatkar),
[Bryan Kwan](https://www.linkedin.com/in/kwan-bryan),
[Jia-Chen Gu](https://jasonforjoy.github.io/index.html),
[Nanyun Peng](https://vnpeng.net/), and
[Kai-Wei Chang](https://kwchang.net/)



This is the official LongMemEval-V2 repository. It contains the public
evaluation harness, data preparation tools, leaderboard packaging utilities,
and the memory baselines reported with the benchmark.

## Overview

LongMemEval-V2 evaluates whether memory systems can help agents acquire the
experience needed to become knowledgeable colleagues in customized
environments. The benchmark pairs manually curated questions with long
histories of multimodal web-agent trajectories. A memory system consumes the
trajectory history and returns compact evidence for downstream question
answering; evaluation targets both answer accuracy and query latency.

LongMemEval-V2 contains:

- 451 manually curated questions.
- 5 memory abilities.
- Up to 500 trajectories per haystack.
- Up to 115M tokens in the largest haystacks.
- Two domains: web and enterprise.
- Two public leaderboard tiers: small and medium.

The benchmark tests five core memory abilities:

- **Static state recall**: remembers important landmarks, page layouts, module
  affordances, and subtle state differences.
- **Dynamic state tracking**: understands how states and actions change the
  environment over time.
- **Workflow knowledge**: knows the steps needed to complete recurring tasks in
  customized environments.
- **Environment gotchas**: recognizes recurring local failure modes and avoids
  environment-specific traps.
- **Premise awareness**: detects assumptions that are valid elsewhere but wrong
  in the current deployment.

## Repository Layout

```text
data/                 download, preparation, and validation scripts
evaluation/           evaluation runner, scoring code, configs, and shell wrappers
leaderboard/          metric merging, LAFS scoring, and submission packaging
memory_modules/       memory backend implementations
```

The repository implements the following memory modules:

- `no_retrieval`: no memory context.
- `rag_query_to_slice`: RAG query to raw state slices.
- `rag_query_to_slice_notes`: RAG query to raw state slices plus trajectory
  notes.
- `agentrunbook_r`: AgentRunbook-R.
- `codex`: vanilla Codex coding-agent memory baseline.
- `agentrunbook_c`: AgentRunbook-C.

## Setup: Environment

LongMemEval-V2 uses Python 3.11. The default conda environment installs
PyTorch through `requirements-torch.txt`. For CUDA 12.4 machines, the torch
install command is:

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Create the environment and install the package:

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate lme-v2-release
pip install -e .
```

Researchers using a different CUDA or CPU setup should install the appropriate
PyTorch build first, either with a direct `pip install` command or by editing
`requirements-torch.txt` before creating the environment.

The environment does not include vLLM. Start or forward your own
OpenAI-compatible model servers, then point the scripts to them. The paper runs
use Qwen3.5-9B as the fixed reader and Qwen3-Embedding-8B for embedding-based
methods. For `codex` and `agentrunbook_c`, download Codex v0.117.0 separately
and set `CODEX_BINARY`.

## Setup: Data

Download and prepare:

```bash
python data/download_data.py --data-root data/longmemeval-v2
export DATA_ROOT="$(pwd)/data/longmemeval-v2"
python data/prepare_data.py --data-root "$DATA_ROOT" --mode symlink
python data/validate_data.py --data-root "$DATA_ROOT" --tier small
```

The default dataset repository is
`xiaowu0162/longmemeval-v2`. Screenshot bundles are stored as `.tar.gz`
archives under `trajectory_screenshots/`; `prepare_data.py` extracts them when
needed and links the resulting directories into:

```text
screenshots/<trajectory_id>/<step>.png
```

## Setup: Model Endpoints and Software

Example endpoint settings:

```bash
# for all experiments
export READER_BASE_URL=http://localhost:8023/v1
export READER_MODEL=Qwen/Qwen3.5-9B

# additionally for RAG and AgentRunbook-R
export LME_CONTROLLER_BASE_URL=http://localhost:8023/v1
export LME_CONTROLLER_MODEL=Qwen/Qwen3.5-9B
export LME_EMBEDDING_BASE_URL=http://localhost:8114/v1
export LME_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

Set for LLM judge (default `gpt-5.2` with `medium` reasoning):

```bash
export OPENAI_API_KEY=...
```

For Codex and AgentRunbook-C:

```bash
export CODEX_BINARY=/path/to/codex-binary
export CODEX_MODEL=gpt-5.4-mini
export CODEX_REASONING_EFFORT=xhigh
```

Codex also expects common command-line tools such as `rg` and `find`.

## Reproducing Baselines

Each shell script accepts extra argparse flags after the environment variables:

```bash
export DATA_ROOT=/path/to/longmemeval-v2
export OUTPUT_ROOT=runs
export TIER=small

evaluation/scripts/run_no_retrieval.sh
evaluation/scripts/run_rag_query_to_slice.sh
evaluation/scripts/run_rag_query_to_slice_notes.sh
evaluation/scripts/run_agentrunbook_r.sh
evaluation/scripts/run_codex.sh
evaluation/scripts/run_agentrunbook_c.sh
```

Each script runs both the web and enterprise domains for the selected tier, writing
outputs such as `runs/no_retrieval_web_small` and
`runs/no_retrieval_enterprise_small`. Set `TIER=medium` to run LME-V2-Medium.

Each run writes `aggregated_metrics.json`. To combine matching enterprise and web runs for the same method and tier:

```bash
python leaderboard/combine_aggregated_metrics.py \
  runs/agentrunbook_r_enterprise_small/aggregated_metrics.json \
  runs/agentrunbook_r_web_small/aggregated_metrics.json \
  -o runs/agentrunbook_r_small_combined_metrics.json
```

## Implementing Your Method

Memory backends inherit from `memory_modules.memory.Memory`. For a minimal
example, see `memory_modules/no_retrieval.py`; for indexed retrieval examples,
see `memory_modules/rag.py` and `memory_modules/agentrunbook_r.py`.

A backend should:

- decorate the class with `@register_memory`;
- set a unique `memory_type`;
- implement `insert(self, trajectory)`, which receives each full trajectory
  object selected for the current haystack;
- implement `query(self, query, query_image=None)`, which receives the question
  text and optional question screenshot path.

`query` must return a list of memory context items:

```python
[
    {"type": "text", "value": "retrieved notes or evidence"},
    {"type": "image", "value": "/absolute/or/relative/path/to/image.png"},
]
```

Text values must be non-empty strings. Image values must point to existing
files. The harness appends these items to the reader prompt and enforces
`--memory-context-max-tokens` before calling the answer model.

During `query`, the backend can call `self.get_query_context()` to access
`question_id`, `question_type`, and the raw question item. Optional hooks include
`post_query_hook(...)` for per-query metadata and `_save_backend(...)` /
`_load_backend(...)` for persisted memory state.

To run a new backend directly, create a memory config JSON:

```json
{
  "memory_type": "your_memory_type",
  "memory_params": {}
}
```

Then pass it to `evaluation/harness.py` with `--memory-config-path`. To expose
the method through `evaluation/run_eval.py` and the shell wrappers, add the
method name and config construction there as well.

## Submitting to Leaderboard

Leaderboard entries measure how much a memory system improves the released
baseline + AgentRunbook accuracy-latency frontier. The score is LAFS gain over
the fixed reference frontier, and a submission may include multiple latency
operating points for the same method and tier.

See [leaderboard/README.md](leaderboard/README.md) for the full packaging
instructions.

Submit leaderboard packages through the
[submission form](https://forms.gle/rxUpiuRKDERqpqSi9). Please do not submit
leaderboard entries as GitHub issues. Informal submission issues will be closed
or deleted.

## Citation


```bibtex
@article{wu2026longmemevalv2,
      title={LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues}, 
      author={Di Wu and Zixiang Ji and Asmi Kawatkar and Bryan Kwan and Jia-Chen Gu and Nanyun Peng and Kai-Wei Chang},
      year={2026},
      eprint={2605.12493},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.12493}, 
}
