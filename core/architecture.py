"""
HIERARCHICAL MAMBA + TRANSFORMER DECODER ML SYSTEM
Plug-and-Play Architecture with Auto-Correction via Reflector
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging

# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class DataType(Enum):
    """Supported input data types"""
    IMAGE = "image"
    TEXT = "text"
    STATISTICAL = "statistical"
    AUDIO = "audio"
    VIDEO = "video"
    CUSTOM = "custom"


class ComponentType(Enum):
    """Module component types"""
    FEEDER = "feeder"
    ENCODER = "encoder"
    DECODER = "decoder"
    REFLECTOR = "reflector"
    TRAINER = "trainer"
    INFERENCE = "inference"


@dataclass
class ModuleConfig:
    """Configuration for any module"""
    name: str
    component_type: ComponentType
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    input_types: List[DataType] = field(default_factory=list)
    output_type: Optional[DataType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'component_type': self.component_type.value,
            'enabled': self.enabled,
            'params': self.params,
            'input_types': [dt.value for dt in self.input_types],
            'output_type': self.output_type.value if self.output_type else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModuleConfig':
        data['component_type'] = ComponentType(data['component_type'])
        data['input_types'] = [DataType(dt) for dt in data['input_types']]
        if data['output_type']:
            data['output_type'] = DataType(data['output_type'])
        return cls(**data)


# ============================================================================
# ABSTRACT BASE MODULES
# ============================================================================

class BaseModule(ABC):
    """Abstract base for all system modules"""
    
    def __init__(self, config: ModuleConfig):
        self.config = config
        self.logger = logging.getLogger(config.name)
        self._initialized = False
        self._lock = None
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize module resources"""
        self._initialized = True
        
    @abstractmethod
    def forward(self, data: Any) -> Any:
        """Process data through module"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status"""
        return {
            'name': self.config.name,
            'initialized': self._initialized,
            'component_type': self.config.component_type.value,
            'enabled': self.config.enabled
        }
    
    def shutdown(self) -> None:
        """Cleanup module resources"""
        self._initialized = False


class DataFeeder(BaseModule):
    """Abstract data input/feeder module"""
    
    @abstractmethod
    def load_batch(self, batch_size: int, **kwargs) -> Tuple[Any, Dict]:
        """Load batch of data"""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate incoming data format"""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess raw data"""
        pass


class Encoder(BaseModule):
    """Abstract encoder module (Hierarchical Mamba)"""
    
    @abstractmethod
    def encode(self, data: Any) -> Any:
        """Encode input to latent representation"""
        pass


class Decoder(BaseModule):
    """Abstract decoder module (Transformer-based)"""
    
    @abstractmethod
    def decode(self, latent: Any) -> Any:
        """Decode from latent representation"""
        pass


class Reflector(BaseModule):
    """Auto-correction module - validates and corrects output"""
    
    @abstractmethod
    def reflect(self, output: Any, ground_truth: Optional[Any] = None) -> Tuple[Any, Dict]:
        """
        Validate output and suggest corrections
        Returns: (corrected_output, reflection_metadata)
        """
        pass
    
    @abstractmethod
    def get_confidence_score(self, output: Any) -> float:
        """Get confidence in output correctness"""
        pass


class Trainer(BaseModule):
    """Training module with integrated reflector feedback"""
    
    @abstractmethod
    def train_step(self, batch: Any, labels: Any) -> Dict[str, float]:
        """Single training step"""
        pass
    
    @abstractmethod
    def validate(self, val_data: Any, val_labels: Any) -> Dict[str, float]:
        """Validation step"""
        pass
    
    @abstractmethod
    def get_reflector_loss(self, reflector: Reflector, 
                          output: Any, target: Any) -> float:
        """Get loss from reflector feedback"""
        pass


class SelfTransformer(BaseModule):
    """Self-upgrade module - improves via internet/external LLMs"""
    
    @abstractmethod
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance"""
        pass
    
    @abstractmethod
    def fetch_improvements(self, source: str = 'internet') -> List[Dict]:
        """Fetch improvements from external sources"""
        pass
    
    @abstractmethod
    def apply_upgrade(self, upgrade_config: Dict) -> bool:
        """Apply improvements to system"""
        pass


# ============================================================================
# MODULAR SYSTEM ORCHESTRATOR
# ============================================================================

class MLSystemOrchestrator:
    """Main system that orchestrates all modules"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger('MLSystemOrchestrator')
        self.modules: Dict[str, BaseModule] = {}
        self.pipeline_sequence: List[str] = []
        self.config_path = config_path
        self.system_config: Dict[str, Any] = {}
        self._threads: Dict[str, Any] = {}
        
    def register_module(self, module: BaseModule) -> None:
        """Register a module in the system"""
        module.initialize()
        self.modules[module.config.name] = module
        self.logger.info(f"Registered module: {module.config.name}")
        
    def set_pipeline(self, sequence: List[str]) -> None:
        """Define execution pipeline"""
        for name in sequence:
            if name not in self.modules:
                raise ValueError(f"Module {name} not registered")
        self.pipeline_sequence = sequence
        self.logger.info(f"Pipeline set: {' -> '.join(sequence)}")
        
    def execute_pipeline(self, data: Any, parallel: bool = False) -> Dict[str, Any]:
        """Execute full pipeline"""
        results = {'input': data, 'stages': {}}
        current_data = data
        
        if parallel:
            return self._execute_parallel(current_data)
        
        for stage_name in self.pipeline_sequence:
            module = self.modules[stage_name]
            if not module.config.enabled:
                continue
                
            current_data = module.forward(current_data)
            results['stages'][stage_name] = current_data
            
        results['output'] = current_data
        return results
    
    def _execute_parallel(self, data: Any) -> Dict[str, Any]:
        """Parallel execution using multithreading"""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        results = {'stages': {}}
        
        with ThreadPoolExecutor(max_workers=len(self.pipeline_sequence)) as executor:
            futures = {}
            current_data = data
            
            for stage_name in self.pipeline_sequence:
                module = self.modules[stage_name]
                if not module.config.enabled:
                    continue
                    
                future = executor.submit(module.forward, current_data)
                futures[stage_name] = future
            
            for stage_name, future in futures.items():
                results['stages'][stage_name] = future.result()
        
        return results
    
    def save_config(self, path: str) -> None:
        """Save system configuration"""
        config_data = {
            'modules': {name: module.config.to_dict() 
                       for name, module in self.modules.items()},
            'pipeline': self.pipeline_sequence,
            'system_config': self.system_config
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
        self.logger.info(f"Config saved to {path}")
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """Load system configuration"""
        with open(path, 'r') as f:
            config_data = json.load(f)
        return config_data
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get full system status"""
        return {
            'modules': {name: module.get_status() 
                       for name, module in self.modules.items()},
            'pipeline': self.pipeline_sequence,
            'total_modules': len(self.modules)
        }
    
    def shutdown(self) -> None:
        """Shutdown all modules"""
        for module in self.modules.values():
            module.shutdown()
        self.logger.info("System shutdown complete")


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_file: Optional[str] = None) -> None:
    """Setup logging for the system"""
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or 'mlsystem.log')
        ]
    )
