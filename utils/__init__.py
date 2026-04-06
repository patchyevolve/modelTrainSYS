"""
Utility Modules
Exports data classifier, inference engine, and upgrade systems.
"""

from utils.data_classifier import (
    DataClassifier,
    DataType,
    TaskType,
    TrainerType,
    FileInfo,
    ContentAnalyzer,
    classify,
    auto_load,
    print_classification,
)
from utils.inference import (
    load_checkpoint,
    run_inference,
    save_results,
    print_report,
)
from utils.project_context import (
    ProjectFileDB,
    ProjectAnalyzer,
    GroqClientCached,
)
from utils.smart_upgrade import (
    SmartUpgradeSystem,
    CodeVerifier,
)
