#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUESTION_TYPES = {"dynamic-environment", "dynamic-environment-abs"}
QUESTION_TYPE_LABEL = "dynamic-full"
DOMAINS = ("web", "enterprise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan resumable one-question dynamic-full runs."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dynamic_full_ids(data_root: Path) -> dict[str, list[str]]:
    ids_by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    questions_path = data_root / "questions.jsonl"
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["question_type"] not in QUESTION_TYPES:
            continue
        domain = row["domain"]
        if domain not in ids_by_domain:
            raise RuntimeError(f"Unsupported dynamic-full domain for {row['id']}: {domain}")
        ids_by_domain[domain].append(row["id"])

    total = sum(len(ids) for ids in ids_by_domain.values())
    if total != 127:
        raise RuntimeError(f"Expected 127 dynamic-full questions, found {total}.")
    return ids_by_domain


def question_output_dir(output_root: Path, domain: str, question_id: str) -> Path:
    return output_root / "items" / domain / question_id


def question_is_complete(output_root: Path, domain: str, question_id: str) -> bool:
    per_question_path = question_output_dir(output_root, domain, question_id) / "per_question.jsonl"
    for row in load_jsonl(per_question_path):
        if row.get("question_id") == question_id:
            return True
    return False


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ids_by_domain = dynamic_full_ids(data_root)
    result: dict[str, Any] = {"output_root": str(output_root)}
    for domain, question_ids in ids_by_domain.items():
        complete = [
            qid for qid in question_ids
            if question_is_complete(output_root, domain, qid)
        ]
        missing = [qid for qid in question_ids if qid not in set(complete)]
        write_lines(output_root / f"{domain}_{QUESTION_TYPE_LABEL}_question_ids.txt", question_ids)
        write_lines(output_root / f"{domain}_{QUESTION_TYPE_LABEL}_complete_question_ids.txt", complete)
        write_lines(output_root / f"{domain}_{QUESTION_TYPE_LABEL}_missing_question_ids.txt", missing)
        result[f"{domain}_total"] = len(question_ids)
        result[f"{domain}_complete"] = len(complete)
        result[f"{domain}_missing"] = len(missing)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
