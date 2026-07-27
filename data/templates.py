from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

log = logging.getLogger("Templates")

TEMPLATE_REGISTRY: Dict[str, "ReasoningTemplate"] = {}


@dataclass
class Segment:
    name: str
    loss_weight: float
    prefix: str
    source: str = "auto"


@dataclass
class ReasoningTemplate:
    name: str
    description: str
    segments: List[Segment]
    separator: str = "\n"
    field_map: Dict[str, str] = field(default_factory=dict)


# ── Built-in templates ──────────────────────────────────────────────────────

_STEP_BY_STEP = ReasoningTemplate(
    name="step_by_step",
    description="Query → Reasoning → Answer with loss masking on query. Best for general reasoning.",
    segments=[
        Segment("query", 0.0, "Query: "),
        Segment("reasoning", 1.0, "Reasoning: "),
        Segment("answer", 1.0, "Answer: "),
    ],
    field_map={
        "question": "query", "questions": "query", "problem": "query",
        "steps": "reasoning", "analysis": "reasoning", "thought": "reasoning",
        "answer": "answer", "result": "answer", "conclusion": "answer",
        "solution": "answer", "output": "answer",
    },
)

_QA = ReasoningTemplate(
    name="qa",
    description="Simple question → answer. Query is masked.",
    segments=[
        Segment("question", 0.0, "Question: "),
        Segment("answer", 1.0, "Answer: "),
    ],
    field_map={
        "question": "question", "questions": "question", "problem": "question",
        "answer": "answer", "response": "answer", "reply": "answer",
    },
)

_ANALYSIS = ReasoningTemplate(
    name="analysis",
    description="Context → Analysis → Conclusion. Use for structured domain data.",
    segments=[
        Segment("context", 0.0, "Context: "),
        Segment("analysis", 1.0, "Analysis: "),
        Segment("conclusion", 1.0, "Conclusion: "),
    ],
    field_map={
        "context": "context", "background": "context", "situation": "context",
        "analysis": "analysis", "evaluation": "analysis", "discussion": "analysis",
        "conclusion": "conclusion", "summary": "conclusion", "result": "conclusion",
    },
)

_CHAT = ReasoningTemplate(
    name="chat",
    description="User → Assistant turn format. User messages are masked.",
    segments=[
        Segment("user", 0.0, "User: "),
        Segment("assistant", 1.0, "Assistant: "),
    ],
    field_map={
        "user": "user", "human": "user", "question": "user",
        "assistant": "assistant", "gpt": "assistant", "response": "assistant",
    },
)

_RAW = ReasoningTemplate(
    name="raw",
    description="No masking, no prefix. Standard language model training.",
    segments=[
        Segment("all", 1.0, ""),
    ],
)

_BUILTIN_TEMPLATES: Dict[str, ReasoningTemplate] = {
    "step_by_step": _STEP_BY_STEP,
    "qa": _QA,
    "analysis": _ANALYSIS,
    "chat": _CHAT,
    "raw": _RAW,
}


# ── Custom template validator ───────────────────────────────────────────────

def _validate_custom_template(data: dict, path: str) -> Optional[ReasoningTemplate]:
    errors: List[str] = []

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("`name` must be a non-empty string")
    if name and (not name.replace("_", "").isalnum()):
        errors.append("`name` must contain only alphanumeric characters and underscores")

    description = data.get("description", "")
    if not isinstance(description, str):
        errors.append("`description` must be a string")

    segments_data = data.get("segments", [])
    if not isinstance(segments_data, list) or len(segments_data) < 1:
        errors.append("`segments` must be a list with at least 1 element")
    elif len(segments_data) > 5:
        errors.append("`segments` must have at most 5 elements")

    segments: List[Segment] = []
    if isinstance(segments_data, list):
        for i, s in enumerate(segments_data):
            s_name = s.get("name") if isinstance(s, dict) else None
            if not isinstance(s_name, str) or not s_name.strip():
                errors.append(f"segments[{i}].name must be a non-empty string")
            lw = s.get("loss_weight") if isinstance(s, dict) else None
            if not isinstance(lw, (int, float)) or not (0.0 <= lw <= 1.0):
                errors.append(f"segments[{i}].loss_weight must be a float in [0.0, 1.0]")
            prefix = s.get("prefix") if isinstance(s, dict) else ""
            if not isinstance(prefix, str):
                errors.append(f"segments[{i}].prefix must be a string")
            if s_name and isinstance(lw, (int, float)):
                segments.append(Segment(
                    name=str(s_name).strip(),
                    loss_weight=float(lw),
                    prefix=str(prefix),
                    source=s.get("source", "auto") if isinstance(s, dict) else "auto",
                ))

    has_trainable = any(s.loss_weight > 0 for s in segments)
    if not has_trainable:
        errors.append("At least one segment must have loss_weight > 0")

    field_map = data.get("field_map", {})
    if not isinstance(field_map, dict):
        errors.append("`field_map` must be a JSON object (dict)")

    if errors:
        for e in errors:
            log.warning(f"  template '{path}': {e}")
        return None

    return ReasoningTemplate(
        name=str(name).strip(),
        description=str(description),
        segments=segments,
        separator=data.get("separator", "\n"),
        field_map={str(k): str(v) for k, v in field_map.items()} if isinstance(field_map, dict) else {},
    )


# ── Registry functions ──────────────────────────────────────────────────────

def init_templates(template_dir: str = "templates"):
    TEMPLATE_REGISTRY.clear()
    TEMPLATE_REGISTRY.update(_BUILTIN_TEMPLATES)

    d = Path(template_dir)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        log.info(f"Created template directory: {d}")
        return

    loaded = 0
    skipped = 0
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"  template '{f.name}': failed to parse JSON — {e}")
            skipped += 1
            continue

        tmpl = _validate_custom_template(data, f.name)
        if tmpl is None:
            skipped += 1
            continue

        TEMPLATE_REGISTRY[tmpl.name] = tmpl
        loaded += 1
        if tmpl.name in _BUILTIN_TEMPLATES:
            log.info(f"  template '{tmpl.name}' (from {f.name}) overrides built-in")
        else:
            log.info(f"  template '{tmpl.name}' (from {f.name})")

    log.info(f"Templates: {loaded} loaded, {skipped} skipped, {len(TEMPLATE_REGISTRY)} total")


def get_template(name: str) -> Optional[ReasoningTemplate]:
    import copy
    return copy.deepcopy(TEMPLATE_REGISTRY.get(name)) if name in TEMPLATE_REGISTRY else None


def list_templates() -> List[str]:
    return sorted(TEMPLATE_REGISTRY.keys())
