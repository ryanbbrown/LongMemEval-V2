import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypedDict

from .memory import Memory, MemoryConfig, MemoryContextItem, register_memory, require
from .thinharness_corpus import (
    TRAJECTORY_SUMMARY_CONCISE_FILENAME,
    TRAJECTORY_SUMMARY_FULL_FILENAME,
    ThinHarnessCorpusBuilder,
    load_json,
    save_json,
)


MAX_TOTAL_SPAN_STATES = 20
DEFAULT_THINHARNESS_MODEL = "openai:gpt-5.4-mini"
DEFAULT_THINHARNESS_TIMEOUT_SECONDS = 1800.0
DEFAULT_THINHARNESS_MAX_ATTEMPTS = 3
DEFAULT_THINHARNESS_MAX_MODEL_REQUESTS = 10000
DEFAULT_THINHARNESS_MAX_TOOL_CALLS = None
DEFAULT_THINHARNESS_OUTPUT_RETRIES = 1
DEFAULT_THINHARNESS_TOOL_RETRIES = 2
DEFAULT_THINHARNESS_TOOLS = ["read", "search", "jsonl_search", "list", "glob"]
DEFAULT_THINHARNESS_EVIDENCE_MODE = "both"
VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
VALID_EVIDENCE_MODES = {"axtree", "image", "both"}

QUERY_PROMPT_TEMPLATE = """# Task Overview

You are acting as a quick memory retrieval module to provide contexts from agent trajectories collected from a customized web environment for a downstream reader to answer questions specific to that environment.

Question:
{query}

You need to aggregate information from the local `trajectories/` directory. Strictly follow the workflow below to collect the information and provide it to the reader module. Follow the navigation hints and tools in the workflow and do not attempt to re-verify/re-explore/rebuild maps because this task has latency constraints.

Be quick and do not over-explore unless necessary. Work inside the current directory and never explore outside of it.


# Output Requirement

Return final JSON with this exact shape:

{{
  "memory_markdown": "## Support Analysis\\n...\\n\\n## Relevant Procedure and Hint Notes\\n...",
  "trajectory_spans": [
    {{
      "trajectory_id": "00332982",
      "start_state_index": 0,
      "end_state_index": 0
    }}
  ]
}}

Requirements:

- `memory_markdown` should contain the two narrative sections only:
  - `## Support Analysis`: a brief plain language description on where the supporting evidence can be found, pointing to the relevant procedure and the spans. If the evidence contradicts the premise of the question, clearly say that the question's premise is wrong and where it is wrong. This serves as the hint to the downstream reader worker to abstain from answering that question.
  - `## Relevant Procedure and Hint Notes`: relevant task procedures and observations found in the sessions. If no useful procedure notes are available, keep this section minimal.
- `trajectory_spans` must use zero-based inclusive indices.
- Preserve span order by importance.
- If you find no useful evidence, still return valid JSON with minimal `memory_markdown` and `trajectory_spans`.


# Workflow

First, `trajectories/` is already organized in a fixed way. Because this layout is fixed, do not spend time rediscovering the directory tree.

- `trajectories/<trajectory_id>/` contains one full session.
- `trajectories/<trajectory_id>/trajectory.json` is the main file for that session.
- `trajectories/<trajectory_id>/states.jsonl` has one state per line in state order. Use this for exact state and span reads.
- `trajectories/<trajectory_id>/actions.jsonl` has one non-empty action per line.
- `trajectories/trajectory_manifest.jsonl` has one row per trajectory.
- `trajectories/state_index.jsonl` has searchable state rows across all trajectories.
- `trajectories/action_index.jsonl` has searchable action rows across all trajectories.
- `trajectories/TRAJECTORY_SUMMARY_CONCISE.md` has a quick high-level overview of each trajectory.
- `trajectories/TRAJECTORY_SUMMARY_FULL.md` has the detailed thought/action sequence for shortlist selection and exact verification.

Rows in `state_index.jsonl`, `action_index.jsonl`, and per-trajectory JSONL files carry an `action_annotated` field: the raw action with the AXTree element it touches appended, e.g. `click('276')  # [276] button 'Delete Review'`. Prefer `action_annotated` over raw `action` when reading actions.

## Step 1: do a quick triage of the query so that you have an expectation of what to do.

- First classify the question quickly before opening any trajectory in detail. Do not blindly browse.
- For direct lookup questions, find one exact state showing the requested field/value/button/page and prefer a single clean supporting span.
- For comparison questions, find the supporting state from one trajectory per side in the comparison. As long as the state contains the support, stop and return; you do not need to verify further.
- For procedure questions, stay within one workflow family unless the question explicitly asks for a shared pattern across workflows. Do not import a plausible step from a different task just because it looks analogous.

## Step 2: inspecting and collecting trajectories.

- Start from `trajectories/TRAJECTORY_SUMMARY_FULL.md` and shortlist only a few likely trajectories using the goal, start URL, action sequence, and final reward. Prefer trajectories on the exact same product/page/workflow family over merely related ones.
- Use `jsonl_search` on `trajectories/state_index.jsonl` and `trajectories/action_index.jsonl` for broad match-style lookup across trajectories.
- Use `where` filters for targeted JSONL lookup. For a state span, combine a trajectory filter with numeric `state_index` range filters such as `{{"field":"trajectory_id","op":"eq","value":"00332982"}}`, `{{"field":"state_index","op":"gte","value":"6","type":"number"}}`, and `{{"field":"state_index","op":"lte","value":"8","type":"number"}}`.
- Use `field_searches` with `jsonl_search` to extract matching internal lines from large multiline string fields such as `accessibility_tree` or `thought`. The top-level `query` selects candidate JSONL rows first; `field_searches` then searches inside the selected field(s) and returns only matching lines/snippets.
- For exact verification inside a shortlisted trajectory, prefer `jsonl_search` with `where` plus `field_searches` over reading a huge raw state row. Example: `{{"path":"trajectories/00332982/states.jsonl","where":[{{"field":"state_index","op":"gte","value":"6","type":"number"}},{{"field":"state_index","op":"lte","value":"8","type":"number"}}],"fields":{{"state_index":0,"url":200,"action_annotated":300}},"field_searches":[{{"field":"accessibility_tree","query":"Order results by|Add Sort|Access Type","regex":true,"context_lines":0,"max_line_chars":300}}]}}`.
- Use multiple `field_searches` when useful, including searches across different returned fields. Example: `{{"field_searches":[{{"field":"accessibility_tree","query":"Delete Review"}},{{"field":"thought","query":"confirm"}}]}}`.
- Use `jsonl_search` on `trajectories/<trajectory_id>/states.jsonl` for one exact state or a small span inside a shortlisted trajectory.
- Use `read` on `trajectories/<trajectory_id>/states.jsonl` with `offset` and `limit` for a short contiguous span. State index 0 is line 1, so span 6:8 is `offset=7, limit=3`.
- Read `trajectories/<trajectory_id>/trajectory.json` only when you need broader context that the summaries and JSONL state/action rows do not provide.
- Use JSONL sidecars only on shortlisted trajectories when possible. They are for exact verification and targeted lookup, not for broad rediscovery.
- IMPORTANT: avoid broad raw search over full `trajectory.json` files. It is time consuming and gives too much context, defeating the purpose of fast retrieval.
- If the evidence contradicts the question, this might mean that the question's premise is wrong. Say so clearly and preserve the contradicting evidence for the downstream reader.
- If the exact evidence is missing, incomplete, or contradictory, do not extrapolate from numeric progressions, nearby rows, similar buttons, or similar workflows. Preserve the contradiction or uncertainty for the downstream reader.
- Keep the final evidence package small. You only need one span to prove one point. The span size should not be too large; usually no more than 3 states in a span suffices. If you need to mention a procedure across many states, mention it in the analysis rather than returning a large span.

# Final Reminder - Important Rules

- Move fast and prefer targeted exploration. Your job is to deliver the relevant evidence as fast as possible.
- Put the most important evidence first.
- Avoid redundant trajectories when multiple trajectories support the same important information.
- Reject nearby-but-not-exact matches. Do not replace the asked field, row, tab, header, button, or state with a similar neighbor.
- You may emit any number of spans, but the total number of states across all spans must be at most 20.
- Count span size inclusively. For example, states 3-5 count as 3 states toward the budget.
- Do not copy AXTree blocks into the output JSON.
"""


class TrajectorySpanOutput(TypedDict):
    trajectory_id: str
    start_state_index: int
    end_state_index: int


class ThinHarnessOutput(TypedDict):
    memory_markdown: str
    trajectory_spans: list[TrajectorySpanOutput]


@dataclass(frozen=True)
class ThinHarnessRunResult:
    text: str
    output: Any | None = None
    usage: dict[str, Any] | None = None


class ThinHarnessRunner(Protocol):
    def run(
        self,
        *,
        prompt: str,
        root: Path,
        params: dict[str, Any],
        trace_dir: Path,
    ) -> ThinHarnessRunResult:
        ...


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _payload_from_runner_result(result: ThinHarnessRunResult) -> Any:
    if result.output is not None:
        if hasattr(result.output, "model_dump"):
            return result.output.model_dump()
        if hasattr(result.output, "__dict__") and not isinstance(result.output, dict):
            return dict(result.output.__dict__)
        return result.output
    try:
        return json.loads(_strip_json_fence(result.text))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ThinHarness final text was not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc


def validate_memory_module_output_payload(payload: Any) -> dict[str, Any]:
    require(isinstance(payload, dict), "memory output must be a JSON object")
    memory_markdown = payload.get("memory_markdown")
    trajectory_spans = payload.get("trajectory_spans")
    require(isinstance(memory_markdown, str), "memory_markdown must be a string")
    require(isinstance(trajectory_spans, list), "trajectory_spans must be a list")

    normalized_spans: list[dict[str, Any]] = []
    total_span_states = 0
    for idx, item in enumerate(trajectory_spans):
        require(isinstance(item, dict), f"trajectory_spans[{idx}] must be an object")
        trajectory_id = item.get("trajectory_id")
        start_state_index = item.get("start_state_index")
        end_state_index = item.get("end_state_index")
        require(
            isinstance(trajectory_id, str) and trajectory_id.strip(),
            f"trajectory_spans[{idx}].trajectory_id must be a non-empty string",
        )
        require(
            isinstance(start_state_index, int)
            and not isinstance(start_state_index, bool)
            and start_state_index >= 0,
            f"trajectory_spans[{idx}].start_state_index must be an integer >= 0",
        )
        require(
            isinstance(end_state_index, int)
            and not isinstance(end_state_index, bool)
            and end_state_index >= start_state_index,
            f"trajectory_spans[{idx}].end_state_index must be an integer >= start_state_index",
        )
        total_span_states += end_state_index - start_state_index + 1
        require(
            total_span_states <= MAX_TOTAL_SPAN_STATES,
            f"trajectory_spans exceed the total state budget: {total_span_states} > {MAX_TOTAL_SPAN_STATES}",
        )
        normalized_spans.append(
            {
                "trajectory_id": trajectory_id.strip(),
                "start_state_index": start_state_index,
                "end_state_index": end_state_index,
            }
        )

    return {
        "memory_markdown": memory_markdown,
        "trajectory_spans": normalized_spans,
    }


def load_api_key(api_key_env: str, api_key_file: str | None) -> str | None:
    env_value = os.getenv(api_key_env)
    if env_value:
        return env_value
    if api_key_file is None:
        return None
    path = Path(api_key_file)
    require(path.exists(), f"Missing API key file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    require(value, f"Empty API key file: {path}")
    return value


class SdkThinHarnessRunner:
    def run(
        self,
        *,
        prompt: str,
        root: Path,
        params: dict[str, Any],
        trace_dir: Path,
    ) -> ThinHarnessRunResult:
        try:
            from thinharness import Harness, HarnessConfig
        except ImportError as exc:
            raise RuntimeError(
                "thinharness is not installed. Install it with `uv add thinharness` "
                "or an editable local checkout before using memory_type=thinharness."
            ) from exc

        api_key = load_api_key(params["api_key_env"], params.get("api_key_file"))
        config = HarnessConfig(
            root=root,
            model=params["model"],
            base_url=params.get("base_url"),
            api_key=api_key,
            builtin_tools=params["builtin_tools"],
            max_model_requests=params["max_model_requests"],
            max_tool_calls=params["max_tool_calls"],
            request_timeout=int(params["timeout_seconds"]),
            output_type=ThinHarnessOutput,
            output_mode=params["output_mode"],
            output_retries=params["output_retries"],
            tool_retries=params["tool_retries"],
            extra_body=params["extra_body"],
            local_trace_dir=trace_dir / "local_traces",
            output_dir=_safe_trace_output_dir(root, trace_dir),
        )
        harness = Harness(config)
        result = harness.run_sync(prompt)
        usage = {
            "model_requests": result.usage.model_requests,
            "tool_calls": result.usage.tool_calls,
            "cancelled_tool_calls": result.usage.cancelled_tool_calls,
            "output_retries": result.usage.output_retries,
            "tool_retries": dict(result.usage.tool_retries),
        }
        return ThinHarnessRunResult(text=result.text, output=result.output, usage=usage)


def _ensure_string_list(value: Any, *, field_name: str) -> list[str]:
    require(isinstance(value, list), f"{field_name} must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        require(isinstance(item, str) and item.strip(), f"{field_name}[{idx}] must be a non-empty string")
        out.append(item)
    return out


def _format_actions(trajectory: dict[str, Any]) -> str:
    actions = [item for item in trajectory.get("actions", []) if isinstance(item, str) and item.strip()]
    if not actions:
        return "1. <no actions recorded>"
    return "\n".join(f"{idx + 1}. {action}" for idx, action in enumerate(actions))


def _format_state_text(state: dict[str, Any]) -> str:
    action = state.get("action_annotated") or state.get("action")
    action_text = action if isinstance(action, str) and action.strip() else "<none>"
    lines = [
        f"State {state['state_index']} (step {state['step']})",
        f"- URL: {state['url']}",
        f"- Action: {action_text}",
    ]
    thought = state.get("thought")
    if isinstance(thought, str) and thought.strip():
        lines.append(f"- Thought: {thought}")
    screenshot = state.get("screenshot")
    if isinstance(screenshot, str) and screenshot.strip():
        lines.append(f"- Screenshot: {screenshot}")
    lines.extend(["- AXTree:", state["accessibility_tree"]])
    return "\n".join(lines) + "\n"


def _safe_trace_output_dir(root: Path, trace_dir: Path) -> str:
    question_part = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in trace_dir.parent.name)
    attempt_part = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in trace_dir.name)
    return str(root / ".thinharness" / "outputs" / question_part / attempt_part)


@register_memory
class ThinHarnessMemory(Memory):
    memory_type = "thinharness"

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        workspace_dir = memory_params.get("workspace_dir")
        query_trace_dir = memory_params.get("query_trace_dir")
        trajectories_root_dir = memory_params.get("trajectories_root_dir")
        runner = memory_params.get("runner")

        model = memory_params.get("model", DEFAULT_THINHARNESS_MODEL)
        base_url = memory_params.get("base_url")
        api_key_env = memory_params.get("api_key_env", "OPENAI_API_KEY")
        api_key_file = memory_params.get("api_key_file")
        timeout_seconds = memory_params.get("timeout_seconds", DEFAULT_THINHARNESS_TIMEOUT_SECONDS)
        max_attempts = memory_params.get("max_attempts", memory_params.get("max_retries", DEFAULT_THINHARNESS_MAX_ATTEMPTS))
        max_model_requests = memory_params.get("max_model_requests", DEFAULT_THINHARNESS_MAX_MODEL_REQUESTS)
        max_tool_calls = memory_params.get("max_tool_calls", DEFAULT_THINHARNESS_MAX_TOOL_CALLS)
        output_retries = memory_params.get("output_retries", DEFAULT_THINHARNESS_OUTPUT_RETRIES)
        tool_retries = memory_params.get("tool_retries", DEFAULT_THINHARNESS_TOOL_RETRIES)
        builtin_tools = memory_params.get("builtin_tools", DEFAULT_THINHARNESS_TOOLS)
        output_mode = memory_params.get("output_mode", "prompted")
        reasoning_effort = memory_params.get("reasoning_effort")
        evidence_mode = memory_params.get("evidence_mode", DEFAULT_THINHARNESS_EVIDENCE_MODE)
        extra_body_obj = memory_params.get("extra_body", {})

        require(isinstance(model, str) and model.strip(), "thinharness model must be a non-empty string")
        require(base_url is None or isinstance(base_url, str), "thinharness base_url must be string or null")
        require(isinstance(api_key_env, str) and api_key_env.strip(), "thinharness api_key_env must be a non-empty string")
        require(api_key_file is None or isinstance(api_key_file, str), "thinharness api_key_file must be string or null")
        require(
            isinstance(timeout_seconds, (int, float))
            and not isinstance(timeout_seconds, bool)
            and float(timeout_seconds) > 0,
            "thinharness timeout_seconds must be a positive number",
        )
        require(
            isinstance(max_attempts, int) and not isinstance(max_attempts, bool) and max_attempts > 0,
            "thinharness max_attempts/max_retries must be a positive integer",
        )
        require(
            isinstance(max_model_requests, int)
            and not isinstance(max_model_requests, bool)
            and max_model_requests > 0,
            "thinharness max_model_requests must be a positive integer",
        )
        require(
            max_tool_calls is None
            or (
                isinstance(max_tool_calls, int)
                and not isinstance(max_tool_calls, bool)
                and max_tool_calls > 0
            ),
            "thinharness max_tool_calls must be null or a positive integer",
        )
        require(
            isinstance(output_retries, int) and not isinstance(output_retries, bool) and output_retries >= 0,
            "thinharness output_retries must be an integer >= 0",
        )
        require(
            isinstance(tool_retries, int) and not isinstance(tool_retries, bool) and tool_retries >= 0,
            "thinharness tool_retries must be an integer >= 0",
        )
        require(output_mode in {"auto", "native", "tool", "prompted", "text"}, "thinharness output_mode is invalid")
        require(
            isinstance(evidence_mode, str) and evidence_mode in VALID_EVIDENCE_MODES,
            "thinharness evidence_mode must be one of: axtree, image, both",
        )
        require(
            reasoning_effort is None
            or (isinstance(reasoning_effort, str) and reasoning_effort in VALID_REASONING_EFFORTS),
            "thinharness reasoning_effort must be one of: none, minimal, low, medium, high, xhigh",
        )
        require(isinstance(extra_body_obj, dict), "thinharness extra_body must be an object")
        if runner is not None:
            require(hasattr(runner, "run"), "thinharness runner must expose run()")

        self.model = model.strip()
        self.base_url = base_url.strip() if isinstance(base_url, str) and base_url.strip() else None
        self.api_key_env = api_key_env.strip()
        self.api_key_file = api_key_file.strip() if isinstance(api_key_file, str) and api_key_file.strip() else None
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.max_model_requests = int(max_model_requests)
        self.max_tool_calls = int(max_tool_calls) if max_tool_calls is not None else None
        self.output_retries = int(output_retries)
        self.tool_retries = int(tool_retries)
        self.builtin_tools = _ensure_string_list(builtin_tools, field_name="thinharness builtin_tools")
        self.output_mode = str(output_mode)
        self.evidence_mode = evidence_mode
        self.reasoning_effort = reasoning_effort
        self.extra_body = dict(extra_body_obj)
        if isinstance(self.reasoning_effort, str):
            self.extra_body = {
                **self.extra_body,
                "reasoning": {
                    **(
                        self.extra_body.get("reasoning")
                        if isinstance(self.extra_body.get("reasoning"), dict)
                        else {}
                    ),
                    "effort": self.reasoning_effort,
                },
            }
        self.workspace_dir = (
            Path(workspace_dir).resolve()
            if isinstance(workspace_dir, str) and workspace_dir.strip()
            else None
        )
        self.trajectories_root_dir = (
            Path(trajectories_root_dir).resolve()
            if isinstance(trajectories_root_dir, str) and trajectories_root_dir.strip()
            else None
        )
        self.query_trace_dir = (
            Path(query_trace_dir).resolve()
            if isinstance(query_trace_dir, str) and query_trace_dir.strip()
            else None
        )
        self.runner: ThinHarnessRunner = runner if runner is not None else SdkThinHarnessRunner()
        self.inserted_trajectory_ids: list[str] = []
        self.inserted_trajectory_id_set: set[str] = set()
        self._attempt_dir_lock = threading.Lock()
        self._insert_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._summaries_dirty = True

        if self.workspace_dir is not None:
            self._ensure_workspace_layout(self.workspace_dir)
        if self.query_trace_dir is not None:
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def memory_config(self) -> MemoryConfig:
        return {
            "memory_type": self.memory_type,
            "memory_params": {
                "model": self.model,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "api_key_file": self.api_key_file,
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_attempts,
                "max_model_requests": self.max_model_requests,
                "max_tool_calls": self.max_tool_calls,
                "output_retries": self.output_retries,
                "tool_retries": self.tool_retries,
                "builtin_tools": list(self.builtin_tools),
                "output_mode": self.output_mode,
                "evidence_mode": self.evidence_mode,
                "reasoning_effort": self.reasoning_effort,
                "extra_body": dict(self.extra_body),
            },
        }

    def configure_runtime(self, **kwargs: object) -> None:
        # The harness passes generation and cancellation kwargs to every backend.
        # ThinHarness owns its model settings through memory_params for this first pass.
        query_trace_dir = kwargs.get("query_trace_dir")
        if query_trace_dir is not None:
            if isinstance(query_trace_dir, Path):
                self.query_trace_dir = query_trace_dir.resolve()
            else:
                require(
                    isinstance(query_trace_dir, str) and query_trace_dir.strip(),
                    "thinharness query_trace_dir runtime override must be a non-empty string or Path",
                )
                self.query_trace_dir = Path(query_trace_dir).resolve()
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)

    def insert(self, trajectory: dict[str, object]) -> None:
        require(self.workspace_dir is not None, "thinharness insert requires workspace_dir")
        with self._insert_lock:
            prepared = ThinHarnessCorpusBuilder(self.workspace_dir, self.trajectories_root_dir).insert(trajectory)
            if prepared.trajectory_id in self.inserted_trajectory_id_set:
                return None
            self.inserted_trajectory_ids.append(prepared.trajectory_id)
            self.inserted_trajectory_id_set.add(prepared.trajectory_id)
            self._write_index_files(self.workspace_dir)
            self._summaries_dirty = True
        return None

    def _ensure_summaries(self) -> None:
        require(self.workspace_dir is not None, "thinharness has no active workspace for summaries")
        trajectory_dir = self.workspace_dir / "trajectories"
        with self._summary_lock:
            files_present = (trajectory_dir / TRAJECTORY_SUMMARY_CONCISE_FILENAME).exists() and (
                trajectory_dir / TRAJECTORY_SUMMARY_FULL_FILENAME
            ).exists()
            if not self._summaries_dirty and files_present:
                return None
            ThinHarnessCorpusBuilder(self.workspace_dir, self.trajectories_root_dir).rebuild_summaries()
            self._summaries_dirty = False
        return None

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        require(isinstance(query, str) and query.strip(), "thinharness query must be a non-empty string")
        require(self.workspace_dir is not None, "thinharness query requires workspace_dir")
        if self.query_trace_dir is None:
            self.query_trace_dir = (self.workspace_dir / "query_traces").resolve()
        self.query_trace_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_summaries()

        query_context = self.get_query_context()
        question_id_value = query_context.get("question_id")
        question_id = question_id_value if isinstance(question_id_value, str) and question_id_value.strip() else "unknown_question"
        last_status = "unknown_failure"
        last_detail: str | None = None
        for attempt_number in range(1, self.max_attempts + 1):
            result = self._run_query_attempt(
                question_id=question_id,
                query=query,
                query_image=query_image,
            )
            if result["success"]:
                return result["memory_context"]
            last_status = result["status"]
            last_detail = result["detail"]
            print(
                (
                    "[thinharness] query attempt failed "
                    f"question_id={question_id} "
                    f"attempt={attempt_number}/{self.max_attempts} "
                    f"status={last_status} "
                    f"detail={last_detail or 'n/a'}"
                ),
                flush=True,
            )
        print(
            (
                "[thinharness] returning empty memory context after "
                f"{self.max_attempts} failed attempts for question_id={question_id} "
                f"last_status={last_status} "
                f"last_detail={last_detail or 'n/a'}"
            ),
            flush=True,
        )
        return []

    def _save_backend(self, output_dir: Path) -> None:
        require(self.workspace_dir is not None, "thinharness has no active workspace to save")
        self._write_index_files(self.workspace_dir)
        self._ensure_summaries()
        if self.workspace_dir.resolve() == output_dir.resolve():
            return None
        for filename in ["index.json", "haystack_manifest.json"]:
            source = self.workspace_dir / filename
            if source.exists():
                shutil.copy2(source, output_dir / filename)
        trajectory_dir = self.workspace_dir / "trajectories"
        if trajectory_dir.exists():
            shutil.copytree(trajectory_dir, output_dir / "trajectories", dirs_exist_ok=True)
        return None

    def _load_backend(self, input_dir: Path) -> None:
        self.workspace_dir = input_dir.resolve()
        self._ensure_workspace_layout(self.workspace_dir)
        index_payload = load_json(self.workspace_dir / "index.json")
        require(isinstance(index_payload, dict), "thinharness index.json must be an object")
        inserted_ids = index_payload.get("inserted_trajectory_ids")
        require(
            isinstance(inserted_ids, list)
            and all(isinstance(item, str) and item for item in inserted_ids),
            "thinharness index.json must contain inserted_trajectory_ids as a list of strings",
        )
        self.inserted_trajectory_ids = list(inserted_ids)
        self.inserted_trajectory_id_set = set(inserted_ids)
        self._summaries_dirty = True
        for trajectory_id in self.inserted_trajectory_ids:
            trajectory_path = self.workspace_dir / "trajectories" / trajectory_id / "trajectory.json"
            require(
                trajectory_path.exists(),
                f"thinharness saved corpus missing trajectory.json for {trajectory_id}: {trajectory_path}",
            )
        return None

    def _ensure_workspace_layout(self, workspace_dir: Path) -> None:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        ThinHarnessCorpusBuilder(workspace_dir).ensure_layout()

    def _write_index_files(self, workspace_dir: Path) -> None:
        save_json(
            workspace_dir / "index.json",
            {
                "memory_type": self.memory_type,
                "updated_at_utc": utc_now_iso(),
                "trajectory_count": len(self.inserted_trajectory_ids),
                "inserted_trajectory_ids": list(self.inserted_trajectory_ids),
            },
        )
        save_json(
            workspace_dir / "haystack_manifest.json",
            {
                "id": f"{self.memory_type}_haystack",
                "variant": "current_memory_workspace",
                "trajectory_ids": list(self.inserted_trajectory_ids),
                "metadata": {
                    "generated_at_utc": utc_now_iso(),
                    "memory_type": self.memory_type,
                    "trajectory_count": len(self.inserted_trajectory_ids),
                    "image_evidence": "available" if self.evidence_mode in {"image", "both"} else "unavailable",
                },
            },
        )

    def _next_attempt_dir(self, question_id: str) -> tuple[int, Path]:
        require(self.query_trace_dir is not None, "thinharness query_trace_dir is not configured")
        with self._attempt_dir_lock:
            question_trace_dir = self.query_trace_dir / question_id
            question_trace_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(
                path
                for path in question_trace_dir.iterdir()
                if path.is_dir() and path.name.startswith("attempt_")
            )
            attempt_index = len(existing) + 1
            attempt_dir = question_trace_dir / f"attempt_{attempt_index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
        return attempt_index, attempt_dir

    def _runner_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_file": self.api_key_file,
            "timeout_seconds": self.timeout_seconds,
            "max_model_requests": self.max_model_requests,
            "max_tool_calls": self.max_tool_calls,
            "output_retries": self.output_retries,
            "tool_retries": self.tool_retries,
            "builtin_tools": list(self.builtin_tools),
            "output_mode": self.output_mode,
            "extra_body": dict(self.extra_body),
        }

    def _run_query_attempt(
        self,
        *,
        question_id: str,
        query: str,
        query_image: str | None,
    ) -> dict[str, Any]:
        require(self.workspace_dir is not None, "thinharness workspace_dir is not configured")
        attempt_index, attempt_dir = self._next_attempt_dir(question_id)
        prompt = QUERY_PROMPT_TEMPLATE.format(query=query)
        if query_image is not None:
            prompt += "\nA question image was supplied, but this text-only backend cannot inspect it.\n"
        prompt_path = attempt_dir / "prompt.md"
        summary_path = attempt_dir / "summary.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        started_at_ts = time.time()
        summary: dict[str, Any] = {
            "question_id": question_id,
            "attempt_index": attempt_index,
            "started_at_utc": datetime.fromtimestamp(started_at_ts, timezone.utc).isoformat(),
            "prompt_path": str(prompt_path),
            "question_has_image": query_image is not None,
            "image_evidence": "unavailable" if query_image is not None else None,
            "runner_params": self._runner_params(),
        }
        try:
            run_result = self.runner.run(
                prompt=prompt,
                root=self.workspace_dir,
                params=self._runner_params(),
                trace_dir=attempt_dir,
            )
            payload = _payload_from_runner_result(run_result)
            normalized_output = self._normalize_output_payload(payload)
            memory_context = self._build_memory_context_from_output(
                normalized_output,
                query_has_image=query_image is not None,
            )
        except Exception as exc:
            summary.update(
                {
                    "completed_at_utc": utc_now_iso(),
                    "duration_seconds": time.time() - started_at_ts,
                    "status_after": "query_error",
                    "status_after_detail": str(exc),
                }
            )
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": "query_error",
                "detail": str(exc),
                "memory_context": [],
            }

        summary.update(
            {
                "completed_at_utc": utc_now_iso(),
                "duration_seconds": time.time() - started_at_ts,
                "status_after": "finished",
                "status_after_detail": None,
                "usage": run_result.usage,
                "agent_output_raw_text": run_result.text,
                "memory_markdown": normalized_output["memory_markdown"],
                "trajectory_spans_raw": normalized_output["trajectory_spans_raw"],
                "trajectory_spans_valid": normalized_output["trajectory_spans_valid"],
                "trajectory_spans_invalid": normalized_output["trajectory_spans_invalid"],
                "memory_context_item_count": len(memory_context),
            }
        )
        save_json(summary_path, summary)
        return {
            "success": True,
            "status": "finished",
            "detail": None,
            "memory_context": memory_context,
        }

    def _load_stored_trajectory(self, trajectory_id: str) -> dict[str, Any] | None:
        require(self.workspace_dir is not None, "thinharness workspace_dir is not configured")
        path = self.workspace_dir / "trajectories" / trajectory_id / "trajectory.json"
        if not path.exists():
            return None
        payload = load_json(path)
        require(isinstance(payload, dict), f"Stored trajectory payload must be an object: {path}")
        states = payload.get("states")
        require(isinstance(states, list), f"Stored trajectory states must be a list: {path}")
        return payload

    def _normalize_output_payload(self, payload: Any) -> dict[str, Any]:
        parsed = validate_memory_module_output_payload(payload)
        normalized: dict[str, Any] = {
            "memory_markdown": parsed["memory_markdown"],
            "trajectory_spans_raw": parsed["trajectory_spans"],
            "trajectory_spans_valid": [],
            "trajectory_spans_invalid": [],
        }
        for span in parsed["trajectory_spans"]:
            trajectory = self._load_stored_trajectory(span["trajectory_id"])
            if trajectory is None:
                normalized["trajectory_spans_invalid"].append({**span, "reason": "unknown_trajectory_id"})
                continue
            states = trajectory.get("states", [])
            if span["end_state_index"] >= len(states):
                normalized["trajectory_spans_invalid"].append(
                    {**span, "reason": f"state_index_out_of_range(len={len(states)})"}
                )
                continue
            normalized["trajectory_spans_valid"].append(span)
        return normalized

    def _build_memory_context_from_output(
        self,
        normalized_output: dict[str, Any],
        *,
        query_has_image: bool = False,
    ) -> list[MemoryContextItem]:
        items: list[MemoryContextItem] = []
        memory_markdown = normalized_output["memory_markdown"]
        valid_spans = normalized_output["trajectory_spans_valid"]

        if isinstance(memory_markdown, str) and memory_markdown.strip():
            items.append({"type": "text", "value": memory_markdown.strip() + "\n"})
        if query_has_image:
            items.append(
                {
                    "type": "text",
                    "value": "## Question Image\nThe benchmark question image is passed directly to the reader outside the memory context.\n",
                }
            )
        if valid_spans:
            span_lines = ["## Trajectory State Spans"]
            for span in valid_spans:
                span_lines.append(
                    f"- {span['trajectory_id']}: states {span['start_state_index']}-{span['end_state_index']}"
                )
            items.append({"type": "text", "value": "\n".join(span_lines) + "\n"})
            items.append({"type": "text", "value": "## Linked Evidence\n"})

        for idx, span in enumerate(valid_spans, start=1):
            trajectory = self._load_stored_trajectory(span["trajectory_id"])
            require(
                trajectory is not None,
                f"Trajectory unexpectedly missing during expansion: {span['trajectory_id']}",
            )
            header = (
                f"### Trajectory span {idx}: {trajectory['id']} states {span['start_state_index']}-{span['end_state_index']}\n\n"
                "Goal\n"
                f"- {trajectory.get('goal', '<goal not found>')}\n\n"
                "Outcome\n"
                f"- {trajectory.get('outcome', '<outcome not found>')}\n\n"
                "Start URL\n"
                f"- {trajectory.get('start_url', '<start url not found>')}\n\n"
                "Actions\n"
                f"{_format_actions(trajectory)}\n\n"
                "Linked state evidence\n"
            )
            items.append({"type": "text", "value": header})
            states = trajectory["states"][span["start_state_index"] : span["end_state_index"] + 1]
            for state in states:
                if self.evidence_mode in {"axtree", "both"}:
                    items.append({"type": "text", "value": _format_state_text(state)})
                if self.evidence_mode in {"image", "both"}:
                    screenshot_value = state.get("screenshot")
                    if isinstance(screenshot_value, str) and screenshot_value.strip():
                        screenshot_path = self.workspace_dir / "trajectories" / trajectory["id"] / screenshot_value
                        require(screenshot_path.exists(), f"Missing ThinHarness screenshot: {screenshot_path}")
                        items.append(
                            {
                                "type": "text",
                                "value": (
                                    "The next image is the screenshot for this exact state. "
                                    "Use it as authoritative evidence for visual-only details such as icons, "
                                    "field prefixes, styling, and banner markers that may be absent from AXTree text.\n"
                                ),
                            }
                        )
                        items.append({"type": "image", "value": str(screenshot_path.resolve())})
        return items
