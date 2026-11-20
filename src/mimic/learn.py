"""CLI: learn a skill from a recording, list skills, or search them.

uv run python -m mimic.learn learn recordings/enter_invoice/recording.json
uv run python -m mimic.learn search "enter supplier bills into accounting"
uv run python -m mimic.learn list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mimic.config import get_settings
from mimic.llm import OllamaClient
from mimic.models import Recording
from mimic.skills import Skill, draft_skill, refine_skill
from mimic.store import SkillStore


def print_skill(skill: Skill) -> None:
    print(f"\nSkill: {skill.name}  (refined by: {skill.refined_by})")
    print(f"  {skill.description}")
    print(f"  Parameters: {', '.join(skill.parameters) or '(none)'}")
    for s in skill.steps:
        target = s.element.label or s.element.text or s.element.selector
        if s.action == "click":
            source = ""
        elif s.parameter:
            source = f"<- {{{s.parameter}}}   (demo: {s.value!r})"
        else:
            source = f"= {s.value!r}   (constant)"
        print(f"  [{s.index:02d}] {s.action:<6} {target:<18} {source}")


def learn(path: Path, use_llm: bool) -> None:
    settings = get_settings()
    recording = Recording.model_validate_json(path.read_text(encoding="utf-8"))
    client = OllamaClient()

    skill = draft_skill(recording, source=str(path))
    if use_llm:
        print(f"Refining with {settings.planner_model}...")
        skill = refine_skill(skill, client, settings.planner_model)

    try:
        embedding = client.embed([skill.embedding_text()])[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Embedding skipped ({type(exc).__name__}). Run: ollama pull nomic-embed-text")
        embedding = None

    skill_id = SkillStore().save(skill, embedding)
    print_skill(skill)
    print(f"\nSaved as skill #{skill_id} ({'with' if embedding else 'without'} embedding).")


def search(query: str) -> None:
    embedding = OllamaClient().embed([query])[0]
    results = SkillStore().search(embedding)
    if not results:
        print("No skills with embeddings yet. Learn one first.")
    for skill, score in results:
        print(f"{score:.3f}  {skill.name}: {skill.description}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mimic skill library")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_learn = sub.add_parser("learn", help="Learn a skill from a recording.json")
    p_learn.add_argument("recording", type=Path)
    p_learn.add_argument("--no-llm", action="store_true", help="Heuristic draft only")
    p_search = sub.add_parser("search", help="Find the skill matching a task description")
    p_search.add_argument("query")
    sub.add_parser("list", help="List learned skills")
    args = parser.parse_args(argv)

    if args.cmd == "learn":
        learn(args.recording, use_llm=not args.no_llm)
    elif args.cmd == "search":
        search(args.query)
    else:
        for name in SkillStore().list_names():
            print(name)


if __name__ == "__main__":
    main()
