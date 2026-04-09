"""
Data Loading Modules
Exports data loaders, datasets, and HuggingFace dataset support.
"""

from data.data_loader import (
    CSVDataset, NumpyDataset, MultiFileDataset, build_loaders
)
from data.text_dataset import (
    CharTokenizer, TextLMDataset, read_text_files, build_text_loaders
)
# torchvision is optional; keep base/text pipelines importable without it.
try:
    from data.image_dataset import (
        ImageFolderDataset, discover_images, get_transforms, build_image_loaders
    )
except ImportError:
    ImageFolderDataset = None
    discover_images = None
    get_transforms = None
    build_image_loaders = None
from data.prefetch_loader import PrefetchLoader
from data.advanced_tokenizer import (
    AdvancedTokenizer,
    ReasoningTokenizer,
)

# Lazy import for HuggingFace datasets (avoid pyarrow compatibility issues)
DATASETS_AVAILABLE = False

def _lazy_load_hf():
    """Lazy load HuggingFace datasets support."""
    global DATASETS_AVAILABLE, build_hf_loaders, load_hf_dataset
    global build_classification_loaders, load_classification_dataset, SUPPORTED_DATASETS, list_supported_datasets
    
    if DATASETS_AVAILABLE:
        return True
    
    try:
        from data import hf_dataset_loader
        build_hf_loaders = hf_dataset_loader.build_hf_loaders
        load_hf_dataset = hf_dataset_loader.load_hf_dataset
        build_classification_loaders = hf_dataset_loader.build_classification_loaders
        load_classification_dataset = hf_dataset_loader.load_classification_dataset
        SUPPORTED_DATASETS = hf_dataset_loader.SUPPORTED_DATASETS
        list_supported_datasets = hf_dataset_loader.list_supported_datasets
        DATASETS_AVAILABLE = hf_dataset_loader.DATASETS_AVAILABLE
        return DATASETS_AVAILABLE
    except (ImportError, AttributeError):
        DATASETS_AVAILABLE = False
        return False
