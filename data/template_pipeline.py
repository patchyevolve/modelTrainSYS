import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .templates import ReasoningTemplate, Segment

log = logging.getLogger("TemplatePipeline")


@dataclass
class StructuredExample:
    segments: Dict[str, str]
    template_name: str = ""
    source_file: str = ""


# ── Content hashing for cache ───────────────────────────────────────────────

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Auto-split for unstructured text ────────────────────────────────────────

def _build_prefix_map(template: ReasoningTemplate) -> List[tuple]:
    """Build (prefix_lower, segment_name) list, longest first."""
    prefixes = []
    for seg in template.segments:
        pfx = seg.prefix.strip()
        if pfx:
            prefixes.append((pfx.lower(), seg.name))
    prefixes.sort(key=lambda x: -len(x[0]))
    return prefixes


def _detect_prefixes(text: str, prefixes: List[tuple]) -> bool:
    """Check if any segment prefix appears at a line start."""
    for line in text.split("\n"):
        stripped = line.strip().lower()
        for pfx, _ in prefixes:
            if stripped.startswith(pfx):
                return True
    return False


def _auto_split_text(text: str, template: ReasoningTemplate) -> List[Dict[str, str]]:
    """
    Split raw text using template segment prefixes as markers.

    E.g. for step_by_step template with prefixes "Question:", "Reasoning:",
    "Answer:" — each prefixed line starts a new segment. Content following
    a prefix (until the next prefix or separator) is assigned to that segment.

    Blank lines and "---" separators split examples.
    """
    prefixes = _build_prefix_map(template)
    seg_names = [s.name for s in template.segments]

    if not prefixes or not _detect_prefixes(text, prefixes):
        # No prefixes found — fall back: all text goes to last trainable segment
        target = None
        for seg in reversed(template.segments):
            if seg.loss_weight > 0:
                target = seg.name
                break
        if target is None:
            target = seg_names[-1]
        paragraphs = [s.strip() for s in text.split("\n") if s.strip()]
        examples = []
        for para in paragraphs:
            if len(para) < 30:
                continue
            examples.append({s.name: "" for s in template.segments} | {target: para})
        return examples

    # Prefix-based parsing
    lines = text.split("\n")
    examples: List[Dict[str, str]] = []
    current = {s: "" for s in seg_names}
    active_seg = seg_names[0] if seg_names else ""

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Separator → new example
        if not stripped or stripped == "---":
            if any(current.values()):
                examples.append(current)
            current = {s: "" for s in seg_names}
            active_seg = seg_names[0] if seg_names else ""
            continue

        # Check for prefix match
        matched = False
        for pfx, seg_name in prefixes:
            if lower.startswith(pfx):
                # Content after prefix
                content = stripped[len(pfx):].strip()
                if current[seg_name]:
                    current[seg_name] += " " + content
                else:
                    current[seg_name] = content
                active_seg = seg_name
                matched = True
                break

        if not matched:
            # Continuation of current segment
            if current[active_seg]:
                current[active_seg] += " " + stripped
            else:
                current[active_seg] = stripped

    if any(current.values()):
        examples.append(current)

    return examples


# ── Structured parser for JSON / JSONL / CSV ────────────────────────────────

def _match_fields(data: dict, template: ReasoningTemplate) -> Optional[Dict[str, str]]:
    """
    Try to map data dict keys to template segments.
    Returns {segment_name: content} or None if no match.
    """
    # Try field_map first (case-insensitive)
    result = {}
    matched = 0

    data_lower = {k.lower(): v for k, v in data.items()}
    field_map_lower = {k.lower(): v for k, v in template.field_map.items()}

    for data_key, data_val in data_lower.items():
        if data_key in field_map_lower:
            seg_name = field_map_lower[data_key]
            if seg_name in {s.name for s in template.segments}:
                result[seg_name] = str(data_val)
                matched += 1

    # If ≥50% of keys matched, use this mapping
    if len(data_lower) > 0 and matched / len(data_lower) >= 0.5:
        return result

    # Fall back: try matching data keys to segment names directly
    result2 = {}
    seg_names = {s.name.lower() for s in template.segments}
    matched2 = 0

    for data_key, data_val in data_lower.items():
        if data_key in seg_names:
            # Find the actual segment name
            for s in template.segments:
                if s.name.lower() == data_key:
                    result2[s.name] = str(data_val)
                    matched2 += 1
                    break

    if len(data_lower) > 0 and matched2 / len(data_lower) >= 0.5:
        return result2

    return result if result else None


def _parse_json_file(path: Path, template: ReasoningTemplate) -> List[Dict[str, str]]:
    """Parse .json or .jsonl file using template field mapping."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    data = json.loads(raw)
    items = data if isinstance(data, list) else [data]

    examples = []
    for item in items:
        if isinstance(item, str):
            # Single string — auto-split
            examples.extend(_auto_split_text(item, template))
        elif isinstance(item, dict):
            mapped = _match_fields(item, template)
            if mapped:
                examples.append(mapped)
            else:
                # Flatten all values into one string and auto-split
                flat = " ".join(str(v) for v in item.values() if isinstance(v, (str, int, float)))
                if flat:
                    examples.extend(_auto_split_text(flat, template))

    return examples


def _parse_jsonl_file(path: Path, template: ReasoningTemplate) -> List[Dict[str, str]]:
    """Parse .jsonl file using template field mapping."""
    examples = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(item, str):
            examples.extend(_auto_split_text(item, template))
        elif isinstance(item, dict):
            mapped = _match_fields(item, template)
            if mapped:
                examples.append(mapped)
            else:
                flat = " ".join(str(v) for v in item.values() if isinstance(v, (str, int, float)))
                if flat:
                    examples.extend(_auto_split_text(flat, template))

    return examples


def _parse_csv_file(path: Path, template: ReasoningTemplate) -> List[Dict[str, str]]:
    """Parse .csv file using template field mapping."""
    import pandas as pd

    df = pd.read_csv(path)
    examples = []

    for _, row in df.iterrows():
        row_dict = {k: str(v) for k, v in row.items() if pd.notna(v)}
        mapped = _match_fields(row_dict, template)
        if mapped:
            examples.append(mapped)
        else:
            # Treat each column as a separate source
            flat = " ".join(row_dict.values())
            if flat:
                examples.extend(_auto_split_text(flat, template))

    return examples


def _parse_txt_file(path: Path, template: ReasoningTemplate) -> List[Dict[str, str]]:
    """Parse .txt file by auto-splitting into segment ratios."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _auto_split_text(text, template)


# ── Cache management ────────────────────────────────────────────────────────

def _cache_path(file_path: str) -> Path:
    return Path(file_path).with_suffix(".reasoning_cache.json")


def _load_cache(cache_file: Path, file_path: str) -> Optional[List[StructuredExample]]:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        stored_hash = data.get("content_hash", "")
        current_hash = _content_hash(Path(file_path).read_text(encoding="utf-8", errors="ignore"))
        if stored_hash != current_hash:
            log.info(f"  cache invalid for {Path(file_path).name} (content changed)")
            return None
        examples = []
        for ex_data in data.get("examples", []):
            examples.append(StructuredExample(
                segments=ex_data.get("segments", {}),
                template_name=ex_data.get("template_name", ""),
                source_file=ex_data.get("source_file", ""),
            ))
        log.info(f"  cache hit for {Path(file_path).name} ({len(examples)} examples)")
        return examples
    except Exception as e:
        log.warning(f"  cache read error for {Path(file_path).name}: {e}")
        return None


def _save_cache(cache_file: Path, file_path: str, template_name: str, examples: List[StructuredExample]):
    try:
        raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        data = {
            "content_hash": _content_hash(raw),
            "template_name": template_name,
            "examples": [
                {
                    "segments": ex.segments,
                    "template_name": ex.template_name,
                    "source_file": ex.source_file,
                }
                for ex in examples
            ],
        }
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info(f"  cached {len(examples)} examples for {Path(file_path).name}")
    except Exception as e:
        log.warning(f"  cache save error for {Path(file_path).name}: {e}")


# ── Main parse entry point ──────────────────────────────────────────────────

_PARSERS = {
    ".json": _parse_json_file,
    ".jsonl": _parse_jsonl_file,
    ".csv": _parse_csv_file,
    ".txt": _parse_txt_file,
}


def parse_examples(
    file_paths: List[str],
    template: ReasoningTemplate,
    use_cache: bool = True,
) -> List[StructuredExample]:
    """Parse all files into StructuredExamples using the given template."""
    all_examples: List[StructuredExample] = []

    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            log.warning(f"  file not found: {fp}")
            continue

        ext = p.suffix.lower()
        parser = _PARSERS.get(ext)
        if parser is None:
            log.warning(f"  unsupported file type: {ext} ({fp})")
            continue

        # Check cache
        if use_cache:
            cache_file = _cache_path(fp)
            cached = _load_cache(cache_file, fp)
            if cached is not None:
                all_examples.extend(cached)
                continue

        # Parse
        raw_examples = parser(p, template)
        if not raw_examples:
            log.info(f"  no examples from {p.name}")
            continue

        structured = [
            StructuredExample(segments=ex, template_name=template.name, source_file=p.name)
            for ex in raw_examples
        ]

        # Cache
        if use_cache:
            _save_cache(_cache_path(fp), fp, template.name, structured)

        all_examples.extend(structured)
        log.info(f"  {p.name}: {len(structured)} examples")

    return all_examples


# ── Render: StructuredExample → token IDs + labels ──────────────────────────

def render_example(
    example: StructuredExample,
    template: ReasoningTemplate,
    tokenizer,
) -> Tuple[List[int], List[int]]:
    all_ids: List[int] = []
    seg_mask: List[int] = []  # 1=trainable, 0=masked for each position
    n = len(template.segments)

    for i, seg in enumerate(template.segments):
        content = example.segments.get(seg.name, "")
        seg_text = seg.prefix + content

        if i < n - 1:
            seg_text += template.separator

        ids = tokenizer.encode(seg_text)

        if n == 1:
            keep = list(ids)
        elif i == 0:
            keep = ids[:-1]
        elif i == n - 1:
            keep = ids[1:]
        else:
            keep = ids[1:-1]

        all_ids.extend(keep)
        trainable = 1 if seg.loss_weight > 0 else 0
        seg_mask.extend([trainable] * len(keep))

    all_labels: List[int] = []
    for t in range(len(all_ids)):
        if t == len(all_ids) - 1:
            all_labels.append(-100)
        elif seg_mask[t] == 1:
            all_labels.append(all_ids[t + 1])
        else:
            all_labels.append(-100)

    return all_ids, all_labels


def render_examples_to_windows(
    examples: List[StructuredExample],
    template: ReasoningTemplate,
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
) -> List[Tuple[List[int], List[int]]]:
    """
    Render all examples to sliding windows of (input_ids, labels).
    Each example can produce multiple windows.
    """
    if stride is None:
        stride = seq_len // 2

    windows: List[Tuple[List[int], List[int]]] = []

    for ex in examples:
        ids, labels = render_example(ex, template, tokenizer)

        if len(ids) < 2:
            continue

        # Slide windows
        for start in range(0, len(ids), stride):
            end = start + seq_len
            if end > len(ids):
                # Pad last window
                x = ids[start:] + [0] * (seq_len - (len(ids) - start))
                y = labels[start:] + [-100] * (seq_len - (len(labels) - start))
            else:
                x = ids[start:end]
                y = labels[start:end]

            if len(x) != seq_len or len(y) != seq_len:
                continue

            windows.append((x, y))

    return windows
