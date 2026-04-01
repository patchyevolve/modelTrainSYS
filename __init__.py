"""
ML System Package - Setup and Initialization
"""

import sys
from pathlib import Path

# Add core modules to path
core_path = Path(__file__).parent / 'core'
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))

# Package metadata
__version__ = '1.0.0'
__author__ = 'ML System Contributors'
__license__ = 'MIT'

# Import main components
try:
    from .core.architecture import (
        MLSystemOrchestrator,
        ModuleConfig,
        ComponentType,
        DataType,
        BaseModule,
        DataFeeder,
        Encoder,
        Decoder,
        Reflector,
        Trainer,
        SelfTransformer,
        setup_logging
    )
    
    from .core.implementations import (
        ImageFeeder,
        TextFeeder,
        StatisticalFeeder,
        MambaBlock,
        HierarchicalMambaEncoder,
        TransformerDecoder,
        PositionalEncoding
    )
    
    from .core.reflector_trainer import (
        NeuralReflector,
        EnsembleReflector,
        ReflectorIntegratedTrainer,
        ReflectionResult
    )
    
    from .core.auto_upgrade import (
        PerformanceAnalyzer,
        ExternalLLMIntegration,
        ArchitectureModifier,
        AutoUpgradeSystem
    )
    
    from .cybersec.trainer import (
        AttackPatternGenerator,
        CybersecurityTrainer
    )
    
    from .interface.chat import (
        ChatCommand,
        MLChatInterface
    )
    
    # Make all components easily accessible
    __all__ = [
        # Core
        'MLSystemOrchestrator',
        'ModuleConfig',
        'ComponentType',
        'DataType',
        'BaseModule',
        'DataFeeder',
        'Encoder',
        'Decoder',
        'Reflector',
        'Trainer',
        'SelfTransformer',
        'setup_logging',
        # Implementations
        'ImageFeeder',
        'TextFeeder',
        'StatisticalFeeder',
        'MambaBlock',
        'HierarchicalMambaEncoder',
        'TransformerDecoder',
        'PositionalEncoding',
        # Reflector & Trainer
        'NeuralReflector',
        'EnsembleReflector',
        'ReflectorIntegratedTrainer',
        'ReflectionResult',
        # Auto-Upgrade
        'PerformanceAnalyzer',
        'ExternalLLMIntegration',
        'ArchitectureModifier',
        'AutoUpgradeSystem',
        # Cybersecurity
        'AttackPatternGenerator',
        'CybersecurityTrainer',
        # Interface
        'ChatCommand',
        'MLChatInterface',
    ]
    
except ImportError as e:
    print(f"Warning: Could not import all components: {e}")
    __all__ = []


# Version info
def version_info():
    """Display version information"""
    return f"ML System v{__version__}"


# Quick start helper
def create_system(name='default'):
    """Quick system creation"""
    return MLSystemOrchestrator()


def setup_default_system():
    """Setup system with default configuration"""
    from .core.implementations import (
        ImageFeeder, HierarchicalMambaEncoder, 
        TransformerDecoder
    )
    from .core.reflector_trainer import NeuralReflector
    
    system = MLSystemOrchestrator()
    
    # Add default modules
    system.register_module(ImageFeeder(ModuleConfig(
        'image_feeder', ComponentType.FEEDER,
        input_types=[DataType.IMAGE]
    )))
    
    system.register_module(HierarchicalMambaEncoder(ModuleConfig(
        'encoder', ComponentType.ENCODER,
        params={'input_dim': 256, 'hidden_dim': 512, 'num_layers': 3}
    )))
    
    system.register_module(TransformerDecoder(ModuleConfig(
        'decoder', ComponentType.DECODER,
        params={'latent_dim': 512, 'output_dim': 256}
    )))
    
    system.register_module(NeuralReflector(ModuleConfig(
        'reflector', ComponentType.REFLECTOR,
        params={'input_dim': 256, 'hidden_dim': 128}
    )))
    
    system.set_pipeline(['image_feeder', 'encoder', 'decoder', 'reflector'])
    
    return system


if __name__ == '__main__':
    print(version_info())
    print("ML System loaded successfully!")
    print(f"Available components: {len(__all__)}")
