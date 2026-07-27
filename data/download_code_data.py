"""
Download code & technical data optimized for Colab.
Uses source tarballs (no git) + HuggingFace datasets (streaming).

Usage:
    python data/download_code_data.py --target ./data/code_corpus
    python data/download_code_data.py --target ./data/code_corpus --no-hf   # skip HF datasets
"""

import argparse, sys, os, json, shutil, urllib.request, tarfile, zipfile, gzip
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("CodeData")

# Official release tarballs (much smaller than git clones)
SOURCE_TARBALLS = [
    # cpython omitted: ~200MB extracted, too heavy for Colab
    ("https://github.com/curl/curl/releases/download/curl-8_5_0/curl-8.5.0.tar.gz", "curl"),
    ("https://github.com/redis/redis/archive/7.2.4.tar.gz", "redis"),
    ("https://github.com/nginx/nginx/archive/release-1.25.3.tar.gz", "nginx"),
    ("https://github.com/libuv/libuv/archive/v1.48.0.tar.gz", "libuv"),
    ("https://www.sqlite.org/2023/sqlite-src-3440200.zip", "sqlite"),
]

CODE_EXTS = {'.py', '.c', '.h', '.js', '.ts', '.rs', '.go', '.java',
             '.cpp', '.hpp', '.rb', '.php', '.swift', '.sh', '.lua',
             '.cc', '.hh', '.toml', '.yaml', '.yml', '.xml', '.md', '.rst'}

TECH_BOOKS = [
    ("https://www.gutenberg.org/cache/epub/58213/pg58213.txt", "computing_machinery.txt"),
    ("https://www.gutenberg.org/cache/epub/63444/pg63444.txt", "algorithms_logic.txt"),
    ("https://www.gutenberg.org/cache/epub/97/pg97.txt", "flatland.txt"),
    ("https://www.gutenberg.org/cache/epub/5797/pg5797.txt", "scientific_papers.txt"),
    ("https://www.gutenberg.org/cache/epub/22764/pg22764.txt", "origin_of_species.txt"),
    ("https://www.gutenberg.org/cache/epub/1228/pg1228.txt", "relativity.txt"),
]


def download(url: str, dest: Path):
    if dest.exists():
        log.info(f"  cached: {dest.name} ({dest.stat().st_size // 1024} KB)")
        return
    log.info(f"  downloading: {dest.name}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            log.info(f"    -> {len(data)//1024} KB")
    except Exception as e:
        log.warning(f"    FAILED: {e}")


def extract_tarball(path: Path, dest: Path, max_mb: int = 100):
    """Extract source code files from tarball/zip."""
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    count = 0
    max_bytes = max_mb * 1024 * 1024

    try:
        name = path.name
        if name.endswith('.tar.gz') or name.endswith('.tgz'):
            mode = 'r:gz'
            arc = tarfile.open(path, mode)
        elif name.endswith('.tar.xz') or name.endswith('.txz'):
            mode = 'r:xz'
            arc = tarfile.open(path, mode)
        elif name.endswith('.zip'):
            arc = zipfile.ZipFile(path)
        else:
            return

        members = arc.getmembers() if hasattr(arc, 'getmembers') else arc.infolist()
        # Sort to get deterministic extraction
        members = sorted(members, key=lambda m: m.name if hasattr(m, 'name') else str(m))

        for member in members:
            name = member.name if hasattr(member, 'name') else str(member)
            if not name or '/' not in name:
                continue
            ext = Path(name.split('/', 1)[1]).suffix.lower()
            if ext not in CODE_EXTS:
                continue
            if total >= max_bytes:
                break

            try:
                if hasattr(arc, 'extractfile'):
                    f = arc.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                else:
                    data = arc.read(member)

                if b'\x00' in data[:512]:
                    continue

                out = dest / name.split('/', 1)[1]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                total += len(data)
                count += 1
            except Exception:
                continue

        arc.close()
    except Exception as e:
        log.warning(f"  extract error: {e}")

    log.info(f"    -> {count} files, {total // 1024 // 1024} MB")


def load_hf_code(target: Path, max_samples: int = 5000):
    """Load code samples from HuggingFace datasets (streaming)."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.info("  'datasets' not installed. Run: pip install datasets")
        return

    hf_dir = target / "hf_code"
    hf_dir.mkdir(parents=True, exist_ok=True)
    out_file = hf_dir / "code_samples.jsonl"

    if out_file.exists():
        log.info(f"  cached: hf code ({out_file.stat().st_size // 1024} KB)")
        return

    log.info("  loading CodeParrot (streaming, 5000 samples)...")
    try:
        ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                          streaming=True, trust_remote_code=True)
        count = 0
        with open(out_file, "w") as f:
            for i, sample in enumerate(ds):
                if i >= max_samples:
                    break
                code = sample.get("content", "")
                if len(code) > 50:
                    f.write(json.dumps({"text": code}) + "\n")
                    count += 1
        log.info(f"    -> {count} code samples ({out_file.stat().st_size // 1024} KB)")
    except Exception as e:
        log.warning(f"    FAILED: {e}. Try: pip install datasets")


def main():
    parser = argparse.ArgumentParser(description="Download code data (Colab-optimized)")
    parser.add_argument("--target", default="data/code_corpus", help="Output directory")
    parser.add_argument("--max-mb", type=int, default=200,
                        help="Max MB of source code to extract per tarball")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace datasets")
    args = parser.parse_args()

    target = Path(args.target)
    raw_dir = target / "raw"
    code_dir = target / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Technical books
    log.info("\n=== Technical books ===")
    for url, name in TECH_BOOKS:
        download(url, raw_dir / name)

    # Phase 2: Source tarballs
    log.info("\n=== Source tarballs ===")
    for url, name in SOURCE_TARBALLS:
        ext = url.split('.')[-1]
        fname = f"{name}.tar.gz" if ext == 'gz' else f"{name}.tar.xz" if ext == 'xz' else f"{name}.zip"
        dest = raw_dir / fname
        download(url, dest)
        extract_tarball(dest, code_dir, max_mb=args.max_mb)

    # Phase 3: HuggingFace code
    if not args.no_hf:
        log.info("\n=== HuggingFace CodeParrot ===")
        load_hf_code(target, max_samples=5000)

    # Phase 4: Generate file list
    log.info("\n=== Manifest ===")
    all_files = sorted([str(f) for f in raw_dir.glob("*.txt")])
    code_files = sorted([str(f) for f in code_dir.rglob("*") if f.is_file()])
    hf_files = sorted([str(f) for f in (target / "hf_code").rglob("*") if f.is_file()])
    all_paths = all_files + code_files + hf_files

    total_mb = sum(Path(f).stat().st_size for f in all_paths) // 1024 // 1024
    extensions = set(Path(f).suffix for f in all_paths)

    manifest = {
        "total_files": len(all_paths),
        "total_mb": total_mb,
        "books": len(all_files),
        "code_files": len(code_files),
        "hf_samples": len(hf_files),
        "extensions": sorted(extensions),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"  {json.dumps(manifest, indent=2)}")

    list_path = target / "file_list.txt"
    list_path.write_text("\n".join(all_paths))
    log.info(f"\nDone! {len(all_paths)} files, {total_mb} MB")
    log.info(f"Run: python scripts/pipeline.py --data-dir {target} --data-chars 50000000")

    return all_paths


if __name__ == "__main__":
    main()
