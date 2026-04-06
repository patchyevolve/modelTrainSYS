"""
Universal Data Classifier - Auto-detects file types and routes to appropriate loader.

Supported types:
- Tabular: .csv, .npy, .npz, .parquet
- Text: .txt, .json, .jsonl, .xml, .csv (text column)
- Image: .jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp
- Audio: .wav, .mp3, .flac, .ogg (future)
- Video: .mp4, .avi, .mkv, .mov (future)

Auto-detection by:
1. File extension
2. Content analysis (magic bytes, structure)
3. CSV column analysis
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

log = logging.getLogger("DataClassifier")

# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

class DataType(Enum):
    TABULAR = "tabular"       # CSV, NPY, NPZ
    TEXT = "text"             # TXT, JSON, JSONL
    IMAGE = "image"          # JPG, PNG, etc.
    AUDIO = "audio"          # WAV, MP3 (future)
    VIDEO = "video"          # MP4, AVI (future)
    UNKNOWN = "unknown"

class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    REGRESSION = "regression"
    LANGUAGE_MODEL = "language_model"
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_GENERATION_CSV = "text_generation_csv"
    GENERATION = "generation"


class TrainerType(Enum):
    CLASSIFIER = "classifier"
    LANGUAGE_MODEL = "language_model"
    IMAGE_CLASSIFIER = "image_classifier"
    CYBERSECURITY = "cybersecurity"
    REGRESSION = "regression"
    HUGGINGFACE_DATASET = "huggingface_dataset"


# ─────────────────────────────────────────────────────────────────────────────
# FILE SIGNATURES (Magic Bytes)
# ─────────────────────────────────────────────────────────────────────────────

MAGIC_BYTES = {
    b'\x89PNG\r\n\x1a\n': "image",      # PNG
    b'\xff\xd8\xff': "image",            # JPEG
    b'GIF87a': "image",                   # GIF87a
    b'GIF89a': "image",                   # GIF89a
    b'RIFF': "audio",                     # WAV (also could be AVI)
    b'ID3': "audio",                      # MP3
    b'fLaC': "audio",                     # FLAC
    b'OggS': "audio",                     # OGG
    b'\x00\x00\x01\x00': "image",         # ICO
    b'BM': "image",                       # BMP
    b'RIFF': "video",                     # AVI (needs further check)
    b'\x00\x00\x00': "video",             # MP4/MOV (container)
}

# Image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".ico", ".svg"}

# Text extensions
TEXT_EXTS = {".txt", ".json", ".jsonl", ".xml", ".csv", ".md", ".html", ".py", ".js"}

# Tabular extensions
TABULAR_EXTS = {".csv", ".npy", ".npz", ".parquet", ".feather"}

# Audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Video extensions
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}


# ─────────────────────────────────────────────────────────────────────────────
# FILE INFO DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileInfo:
    path: Path
    data_type: DataType
    task_type: TaskType
    size_bytes: int
    extension: str
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    is_binary: bool = True
    class_names: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "data_type": self.data_type.value,
            "task_type": self.task_type.value,
            "size_bytes": self.size_bytes,
            "extension": self.extension,
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "is_binary": self.is_binary,
            "class_names": self.class_names,
            "warnings": self.warnings or [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class ContentAnalyzer:
    """Analyzes file content to determine data type and structure."""

    @staticmethod
    def detect_by_magic(path: Path) -> Optional[str]:
        """Detect file type by reading magic bytes."""
        try:
            with open(path, "rb") as f:
                header = f.read(16)
            
            for magic, file_type in MAGIC_BYTES.items():
                if header.startswith(magic):
                    return file_type
        except Exception as e:
            log.debug(f"Magic byte detection failed for {path}: {e}")
        return None

    @staticmethod
    def analyze_csv(path: Path) -> Dict[str, Any]:
        """Analyze CSV structure to determine task type."""
        import pandas as pd
        
        try:
            df = pd.read_csv(path, nrows=1000)
            
            num_cols = len(df.columns)
            num_rows = len(df)
            
            label_col = None
            label_candidates = [
                "label", "class", "target", "category", "y",
                "attack_detected", "is_attack", "malware",
                "list", "result", "outcome"
            ]
            
            for col in df.columns:
                if col.lower() in label_candidates:
                    label_col = col
                    break
            
            text_col = None
            text_col_candidates = [
                "text", "content", "body", "message", "sentence", 
                "review", "description", "comment", "input", "output",
                "prompt", "response", "article", "document"
            ]
            
            for col in df.columns:
                if col.lower() in text_col_candidates:
                    text_col = col
                    break
            
            text_cols = df.select_dtypes(include="object").columns.tolist()
            
            avg_text_len = 0
            if text_cols:
                sample_texts = df[text_cols[0]].dropna().astype(str)
                if len(sample_texts) > 0:
                    avg_text_len = sample_texts.str.len().mean()
            
            if text_col and avg_text_len > 100:
                is_text_gen = True
                for col in df.columns:
                    if col != text_col and df[col].dtype in ['int64', 'float64']:
                        if df[col].nunique() <= 10:
                            is_text_gen = False
                            break
                
                if is_text_gen:
                    return {
                        "num_samples": num_rows,
                        "num_features": num_cols,
                        "num_classes": None,
                        "is_binary": False,
                        "task_type": TaskType.TEXT_GENERATION_CSV,
                        "label_column": text_col,
                        "class_names": None,
                        "is_text_generation": True,
                    }
            
            if label_col is None and text_col is None:
                if len(text_cols) >= num_cols * 0.7 and avg_text_len > 50:
                    return {
                        "num_samples": num_rows,
                        "num_features": num_cols,
                        "num_classes": None,
                        "is_binary": False,
                        "task_type": TaskType.TEXT_GENERATION_CSV,
                        "label_column": None,
                        "class_names": None,
                        "is_text_generation": True,
                    }
                return {
                    "num_samples": num_rows,
                    "num_features": num_cols,
                    "num_classes": 1,
                    "is_binary": True,
                    "task_type": TaskType.REGRESSION,
                    "label_column": None,
                    "class_names": None,
                }
            
            if label_col is None:
                label_col = df.columns[-1]
            
            label_values = df[label_col].dropna().unique()
            num_classes = len(label_values)
            is_binary = num_classes <= 2
            
            if num_classes == 2:
                task = TaskType.BINARY_CLASSIFICATION
            elif num_classes > 2:
                task = TaskType.MULTI_CLASS_CLASSIFICATION
            else:
                task = TaskType.REGRESSION
            
            return {
                "num_samples": num_rows,
                "num_features": num_cols - 1,
                "num_classes": num_classes,
                "is_binary": is_binary,
                "task_type": task,
                "label_column": label_col,
                "class_names": [str(v) for v in label_values] if num_classes <= 20 else None,
            }
        except Exception as e:
            log.warning(f"CSV analysis failed for {path}: {e}")
            return {
                "num_samples": None,
                "num_features": None,
                "num_classes": None,
                "is_binary": True,
                "task_type": TaskType.BINARY_CLASSIFICATION,
                "label_column": None,
                "class_names": None,
            }

    @staticmethod
    def analyze_npy(path: Path) -> Dict[str, Any]:
        """Analyze NPY/NPZ structure."""
        import numpy as np
        
        try:
            if path.suffix == ".npz":
                data = np.load(path)
                keys = list(data.keys())
                X = data[keys[0]]
                y = data[keys[1]] if len(keys) > 1 else None
            else:
                arr = np.load(path)
                if arr.ndim == 2:
                    X = arr[:, :-1]
                    y = arr[:, -1]
                else:
                    X = arr
                    y = None
            
            num_classes = None
            is_binary = True
            task = TaskType.REGRESSION
            
            if y is not None:
                unique = np.unique(y)
                num_classes = len(unique)
                is_binary = num_classes <= 2
                if num_classes == 2:
                    task = TaskType.BINARY_CLASSIFICATION
                elif num_classes > 2:
                    task = TaskType.MULTI_CLASS_CLASSIFICATION
            
            return {
                "num_samples": len(X),
                "num_features": X.shape[1] if X.ndim > 1 else 1,
                "num_classes": num_classes,
                "is_binary": is_binary,
                "task_type": task,
            }
        except Exception as e:
            log.warning(f"NPY analysis failed for {path}: {e}")
            return {
                "num_samples": None,
                "num_features": None,
                "num_classes": None,
                "is_binary": True,
                "task_type": TaskType.BINARY_CLASSIFICATION,
            }

    @staticmethod
    def analyze_text(path: Path) -> Dict[str, Any]:
        """Analyze text file structure."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(10000)  # Read first 10KB
            
            lines = content.split("\n")
            num_lines = len([l for l in lines if l.strip()])
            
            # Try to detect structure
            if path.suffix == ".jsonl":
                try:
                    first_line = json.loads(lines[0])
                    if isinstance(first_line, dict):
                        # Check for text fields
                        text_fields = ["text", "content", "message", "sentence", "body"]
                        has_text = any(f in first_line for f in text_fields)
                        if has_text:
                            return {
                                "num_samples": num_lines,
                                "task_type": TaskType.LANGUAGE_MODEL,
                            }
                except:
                    pass
            
            # Plain text - treat as corpus
            return {
                "num_samples": num_lines,
                "task_type": TaskType.LANGUAGE_MODEL,
            }
        except Exception as e:
            log.warning(f"Text analysis failed for {path}: {e}")
            return {
                "num_samples": None,
                "task_type": TaskType.LANGUAGE_MODEL,
            }

    @staticmethod
    def analyze_image_folder(paths: List[Path]) -> Dict[str, Any]:
        """Analyze image folder structure for classification."""
        samples = []
        class_counts = {}
        
        for p in paths:
            if p.is_dir():
                # Check subdirectories
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                if subdirs:
                    # Classified folder structure
                    for sd in subdirs:
                        images = [f for f in sd.iterdir() 
                                  if f.suffix.lower() in IMAGE_EXTS]
                        class_counts[sd.name] = len(images)
                        samples.extend(images)
                else:
                    # Flat folder - single class
                    images = [f for f in p.iterdir() 
                              if f.suffix.lower() in IMAGE_EXTS]
                    class_counts[p.name] = len(images)
                    samples.extend(images)
            elif p.suffix.lower() in IMAGE_EXTS:
                samples.append(p)
        
        num_classes = len(class_counts)
        is_binary = num_classes <= 2
        
        return {
            "num_samples": len(samples),
            "num_classes": num_classes,
            "is_binary": is_binary,
            "task_type": TaskType.IMAGE_CLASSIFICATION,
            "class_names": list(class_counts.keys()) if class_counts else None,
            "class_counts": class_counts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class DataClassifier:
    """
    Universal data classifier that auto-detects file types and task types.
    
    Usage:
        classifier = DataClassifier()
        info = classifier.classify_file("path/to/data.csv")
        
        if info.data_type == DataType.TABULAR:
            from data.data_loader import build_loaders
            train_loader, val_loader, data_info = build_loaders(["path/to/data.csv"], batch_size=32)
        elif info.data_type == DataType.IMAGE:
            from data.image_dataset import build_image_loaders
            train_loader, val_loader, class_names, info = build_image_loaders(["path/to/images/"], batch_size=32)
    """

    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self._cache: Dict[str, FileInfo] = {}

    def classify_file(self, path: Union[str, Path]) -> FileInfo:
        """Classify a single file."""
        path = Path(path)
        
        if str(path) in self._cache:
            return self._cache[str(path)]
        
        if not path.exists():
            return FileInfo(
                path=path,
                data_type=DataType.UNKNOWN,
                task_type=TaskType.BINARY_CLASSIFICATION,
                size_bytes=0,
                extension=path.suffix,
                warnings=[f"File not found: {path}"]
            )
        
        ext = path.suffix.lower()
        size = path.stat().st_size
        
        # 1. Try by extension first
        data_type = self._get_type_by_extension(ext)
        task_type = TaskType.BINARY_CLASSIFICATION
        warnings = []
        
        # 2. Try by magic bytes for images
        if data_type == DataType.UNKNOWN:
            magic_type = self.analyzer.detect_by_magic(path)
            if magic_type == "image":
                data_type = DataType.IMAGE
        
        # 3. Analyze content
        num_samples = None
        num_features = None
        num_classes = None
        is_binary = True
        class_names = None
        
        if data_type == DataType.TABULAR or ext == ".csv":
            if ext in (".csv",):
                analysis = self.analyzer.analyze_csv(path)
            elif ext in (".npy", ".npz"):
                analysis = self.analyzer.analyze_npy(path)
            else:
                analysis = {}
            
            num_samples = analysis.get("num_samples")
            num_features = analysis.get("num_features")
            num_classes = analysis.get("num_classes")
            is_binary = analysis.get("is_binary", True)
            task_type = analysis.get("task_type", TaskType.BINARY_CLASSIFICATION)
            class_names = analysis.get("class_names")
            
        elif data_type == DataType.TEXT:
            analysis = self.analyzer.analyze_text(path)
            task_type = analysis.get("task_type", TaskType.LANGUAGE_MODEL)
            num_samples = analysis.get("num_samples")
        
        # Check for warnings
        if num_samples and num_samples < 100:
            warnings.append(f"Small dataset: only {num_samples} samples")
        if num_classes and num_classes == 1:
            warnings.append("Only 1 class found - need at least 2 for classification")
        
        info = FileInfo(
            path=path,
            data_type=data_type,
            task_type=task_type,
            size_bytes=size,
            extension=ext,
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            is_binary=is_binary,
            class_names=class_names,
            warnings=warnings or None,
        )
        
        self._cache[str(path)] = info
        return info

    def classify_files(self, paths: List[Union[str, Path]]) -> List[FileInfo]:
        """Classify multiple files."""
        return [self.classify_file(p) for p in paths]

    def _get_type_by_extension(self, ext: str) -> DataType:
        """Map extension to data type."""
        if ext in TABULAR_EXTS:
            return DataType.TABULAR
        elif ext in IMAGE_EXTS:
            return DataType.IMAGE
        elif ext in TEXT_EXTS:
            return DataType.TEXT
        elif ext in AUDIO_EXTS:
            return DataType.AUDIO
        elif ext in VIDEO_EXTS:
            return DataType.VIDEO
        else:
            return DataType.UNKNOWN

    def auto_load(self, paths: List[Union[str, Path]], 
                  batch_size: int = 32,
                  **kwargs) -> Tuple[Any, Any, Dict]:
        """
        Auto-detect file types and load with appropriate loader.
        
        Returns: (train_loader, val_loader, data_info)
        """
        if not paths:
            raise ValueError("No files provided")
        
        # Classify all files
        infos = self.classify_files(paths)
        
        # Group by data type
        by_type: Dict[DataType, List[Path]] = {}
        for info in infos:
            if info.data_type != DataType.UNKNOWN:
                by_type.setdefault(info.data_type, []).append(info.path)
        
        if not by_type:
            raise ValueError("No recognizable files found")
        
        if len(by_type) > 1:
            log.warning(f"Mixed file types detected: {list(by_type.keys())}")
            # Use the most common type
            data_type = max(by_type.keys(), key=lambda k: len(by_type[k]))
        else:
            data_type = list(by_type.keys())[0]
        
        file_paths = [str(p) for p in by_type[data_type]]
        
        # Load with appropriate loader
        if data_type == DataType.TABULAR:
            from data.data_loader import build_loaders
            train_loader, val_loader, data_info = build_loaders(
                file_paths, batch_size=batch_size, **kwargs)
            return train_loader, val_loader, data_info
        
        elif data_type == DataType.IMAGE:
            from data.image_dataset import build_image_loaders
            train_loader, val_loader, class_names, info = build_image_loaders(
                file_paths, batch_size=batch_size, **kwargs)
            data_info = {
                **info,
                "class_names": class_names,
                "data_type": "image",
            }
            return train_loader, val_loader, data_info
        
        elif data_type == DataType.TEXT:
            from data.text_dataset import build_text_loaders
            train_loader, val_loader, tokenizer, info = build_text_loaders(
                file_paths, batch_size=batch_size, **kwargs)
            data_info = {
                **info,
                "data_type": "text",
            }
            return train_loader, val_loader, data_info
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def print_info(self, info: FileInfo) -> str:
        """Format file info as readable string."""
        lines = [
            f"File: {info.path.name}",
            f"Type: {info.data_type.value}",
            f"Task: {info.task_type.value}",
            f"Size: {info.size_bytes / 1024:.1f} KB",
        ]
        if info.num_samples:
            lines.append(f"Samples: {info.num_samples:,}")
        if info.num_features:
            lines.append(f"Features: {info.num_features}")
        if info.num_classes:
            lines.append(f"Classes: {info.num_classes} ({'binary' if info.is_binary else 'multi-class'})")
        if info.class_names:
            lines.append(f"Class names: {', '.join(str(n) for n in info.class_names[:5])}")
        if info.warnings:
            for w in info.warnings:
                lines.append(f"Warning: {w}")
        return "\n".join(lines)

    def select_trainer(self, info: FileInfo) -> Tuple[TrainerType, str]:
        """
        Select the appropriate trainer based on file analysis.
        
        Returns: (TrainerType, recommended_model_config)
        """
        if info.data_type == DataType.IMAGE:
            return TrainerType.IMAGE_CLASSIFIER, "Image Classification"
        
        if info.data_type == DataType.TEXT:
            return TrainerType.LANGUAGE_MODEL, "Text Generation"
        
        if info.data_type == DataType.TABULAR:
            if info.task_type == TaskType.REGRESSION:
                return TrainerType.REGRESSION, "Regression"
            
            if info.task_type == TaskType.LANGUAGE_MODEL:
                return TrainerType.LANGUAGE_MODEL, "Text Generation from CSV"
            
            if info.num_classes and info.num_classes > 1:
                if info.is_binary:
                    return TrainerType.CLASSIFIER, "Binary Classification"
                else:
                    return TrainerType.CLASSIFIER, "Multi-class Classification"
        
        return TrainerType.CLASSIFIER, "Default Classifier"

    def analyze_and_recommend(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a file and provide training recommendations."""
        info = self.classify_file(path)
        trainer_type, trainer_name = self.select_trainer(info)
        
        recommendations = {
            "file_info": info.to_dict(),
            "recommended_trainer": trainer_type.value,
            "trainer_description": trainer_name,
            "ui_model_type": self._get_ui_model_type(trainer_type),
            "suggestions": [],
        }
        
        if info.num_samples and info.num_samples < 100:
            recommendations["suggestions"].append(
                f"Small dataset: only {info.num_samples} samples. Consider data augmentation."
            )
        
        if info.num_features and info.num_features > 1000:
            recommendations["suggestions"].append(
                f"High dimensionality ({info.num_features} features). Consider dimensionality reduction."
            )
        
        return recommendations

    def _get_ui_model_type(self, trainer_type: TrainerType) -> str:
        """Map trainer type to UI model type."""
        mapping = {
            TrainerType.CLASSIFIER: "Hierarchical Mamba",
            TrainerType.LANGUAGE_MODEL: "Text Generation",
            TrainerType.IMAGE_CLASSIFIER: "Image Classification",
            TrainerType.CYBERSECURITY: "Cybersecurity",
            TrainerType.REGRESSION: "Hierarchical Mamba",
            TrainerType.HUGGINGFACE_DATASET: "Dataset Training",
        }
        return mapping.get(trainer_type, "Hierarchical Mamba")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def classify(path: Union[str, Path]) -> FileInfo:
    """Quick classify a single file."""
    return DataClassifier().classify_file(path)

def auto_load(paths: List[Union[str, Path]], batch_size: int = 32, **kwargs):
    """Quick auto-load multiple files."""
    return DataClassifier().auto_load(paths, batch_size=batch_size, **kwargs)

def print_classification(path: Union[str, Path]) -> str:
    """Quick print classification info."""
    info = classify(path)
    return DataClassifier().print_info(info)


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_classifier.py <file_or_folder>")
        print("\nSupported types:")
        print("  Tabular: csv, npy, npz")
        print("  Text: txt, json, jsonl")
        print("  Image: jpg, png, gif, etc.")
        sys.exit(1)
    
    path = sys.argv[1]
    print_classification(path)
