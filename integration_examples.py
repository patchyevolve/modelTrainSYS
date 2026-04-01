"""
Complete Integration Example - Hierarchical Mamba + Transformer System
Demonstrates plug-and-play architecture with all components
"""

import torch
import torch.nn as nn
from typing import List
import sys
from pathlib import Path

# Simulated imports (would be actual in production)
# from mlsystem.core.architecture import *
# from mlsystem.core.implementations import *
# from mlsystem.core.reflector_trainer import *
# from mlsystem.cybersec.trainer import *
# from mlsystem.core.auto_upgrade import *
# from mlsystem.interface.chat import *


# ============================================================================
# EXAMPLE 1: BASIC IMAGE PROCESSING PIPELINE
# ============================================================================

def example_basic_image_pipeline():
    """Example: Simple image processing with encoder-decoder"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Image Processing Pipeline")
    print("="*70)
    
    # Import architecture components
    from mlsystem.core.architecture import (
        MLSystemOrchestrator, ModuleConfig, ComponentType, DataType
    )
    from mlsystem.core.implementations import (
        ImageFeeder, HierarchicalMambaEncoder, TransformerDecoder
    )
    
    # Create system orchestrator
    system = MLSystemOrchestrator()
    
    # Create and register modules
    feeder_config = ModuleConfig(
        name='image_feeder',
        component_type=ComponentType.FEEDER,
        input_types=[DataType.IMAGE],
        output_type=DataType.IMAGE,
        params={'batch_size': 16}
    )
    feeder = ImageFeeder(feeder_config)
    system.register_module(feeder)
    
    encoder_config = ModuleConfig(
        name='mamba_encoder',
        component_type=ComponentType.ENCODER,
        input_types=[DataType.IMAGE],
        output_type=DataType.IMAGE,
        params={
            'input_dim': 256,
            'hidden_dim': 512,
            'num_layers': 3,
            'num_scales': 3
        }
    )
    encoder = HierarchicalMambaEncoder(encoder_config)
    system.register_module(encoder)
    
    decoder_config = ModuleConfig(
        name='transformer_decoder',
        component_type=ComponentType.DECODER,
        input_types=[DataType.IMAGE],
        output_type=DataType.IMAGE,
        params={
            'latent_dim': 512,
            'output_dim': 256,
            'num_heads': 8,
            'num_layers': 3,
            'ff_dim': 2048
        }
    )
    decoder = TransformerDecoder(decoder_config)
    system.register_module(decoder)
    
    # Set pipeline
    system.set_pipeline(['image_feeder', 'mamba_encoder', 'transformer_decoder'])
    
    # Create dummy image tensor
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Execute pipeline
    print("\nProcessing image...")
    results = system.execute_pipeline(dummy_image, parallel=False)
    
    print(f"✓ Input shape: {results['input'].shape}")
    print(f"✓ Output shape: {results['output'].shape}")
    
    system.shutdown()


# ============================================================================
# EXAMPLE 2: TEXT PROCESSING WITH REFLECTOR
# ============================================================================

def example_text_with_reflector():
    """Example: Text processing with auto-correction reflector"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Text Processing with Reflector Correction")
    print("="*70)
    
    from mlsystem.core.architecture import (
        MLSystemOrchestrator, ModuleConfig, ComponentType, DataType
    )
    from mlsystem.core.implementations import (
        TextFeeder, HierarchicalMambaEncoder, TransformerDecoder
    )
    from mlsystem.core.reflector_trainer import NeuralReflector
    
    # Create system
    system = MLSystemOrchestrator()
    
    # Text feeder
    feeder_config = ModuleConfig(
        name='text_feeder',
        component_type=ComponentType.FEEDER,
        input_types=[DataType.TEXT],
        output_type=DataType.TEXT,
        params={'max_length': 512}
    )
    feeder = TextFeeder(feeder_config)
    system.register_module(feeder)
    
    # Encoder
    encoder_config = ModuleConfig(
        name='text_encoder',
        component_type=ComponentType.ENCODER,
        input_types=[DataType.TEXT],
        output_type=None,
        params={'input_dim': 512, 'hidden_dim': 1024, 'num_layers': 4}
    )
    encoder = HierarchicalMambaEncoder(encoder_config)
    system.register_module(encoder)
    
    # Decoder
    decoder_config = ModuleConfig(
        name='text_decoder',
        component_type=ComponentType.DECODER,
        input_types=[DataType.TEXT],
        output_type=DataType.TEXT,
        params={'latent_dim': 1024, 'output_dim': 512}
    )
    decoder = TransformerDecoder(decoder_config)
    system.register_module(decoder)
    
    # Reflector for auto-correction
    reflector_config = ModuleConfig(
        name='text_reflector',
        component_type=ComponentType.REFLECTOR,
        input_types=[DataType.TEXT],
        params={'input_dim': 512, 'hidden_dim': 256, 'threshold': 0.7}
    )
    reflector = NeuralReflector(reflector_config)
    system.register_module(reflector)
    
    # Set pipeline
    system.set_pipeline(['text_feeder', 'text_encoder', 'text_decoder', 'text_reflector'])
    
    # Process sample text
    sample_text = "This is a test sentence for the ML system"
    print(f"\nProcessing: {sample_text}")
    
    results = system.execute_pipeline(sample_text)
    print(f"✓ Pipeline execution completed")
    print(f"✓ Final output: {results['stages'].get('text_reflector', 'N/A')}")
    
    system.shutdown()


# ============================================================================
# EXAMPLE 3: CYBERSECURITY TRAINING
# ============================================================================

def example_cybersecurity_training():
    """Example: Train model on attack patterns with offense learning"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Cybersecurity Attack Detection Training")
    print("="*70)
    
    from mlsystem.core.architecture import (
        MLSystemOrchestrator, ModuleConfig, ComponentType, DataType
    )
    from mlsystem.core.implementations import StatisticalFeeder
    from mlsystem.cybersec.trainer import CybersecurityTrainer, AttackPatternGenerator
    from mlsystem.core.reflector_trainer import NeuralReflector
    
    # Create simple model for demo
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    
    system = MLSystemOrchestrator()
    
    # Feeder
    feeder_config = ModuleConfig(
        name='traffic_feeder',
        component_type=ComponentType.FEEDER,
        input_types=[DataType.STATISTICAL],
        output_type=DataType.STATISTICAL,
    )
    feeder = StatisticalFeeder(feeder_config)
    system.register_module(feeder)
    
    # Reflector
    reflector_config = ModuleConfig(
        name='attack_reflector',
        component_type=ComponentType.REFLECTOR,
        params={'input_dim': 4, 'hidden_dim': 32}
    )
    reflector = NeuralReflector(reflector_config)
    system.register_module(reflector)
    
    # Cybersecurity trainer
    trainer_config = ModuleConfig(
        name='cybersec_trainer',
        component_type=ComponentType.TRAINER,
        params={
            'model': model,
            'reflector': reflector,
            'optimizer': 'adam',
            'lr': 1e-3,
            'reflector_weight': 0.3
        }
    )
    trainer = CybersecurityTrainer(trainer_config)
    system.register_module(trainer)
    
    # Generate and train on attack patterns
    print("\nGenerating attack patterns...")
    attack_gen = AttackPatternGenerator()
    
    # Train on mixed data
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")
        
        # Attack data
        attacks = attack_gen.generate_attack_batch(16)
        attack_features, attack_labels = trainer.generate_training_data(16)
        
        # Benign data
        benign_features, benign_labels = trainer.generate_benign_data(16)
        
        # Combine
        all_features = torch.cat([attack_features, benign_features], dim=0)
        all_labels = torch.cat([attack_labels, benign_labels], dim=0)
        
        # Train step
        metrics = trainer.train_step_cybersec(all_features, all_labels)
        
        print(f"  Loss: {metrics['total_loss']:.4f}")
        print(f"  Attack detection rate: {metrics.get('attack_detection_rate', 'N/A')}")
    
    # Generate defense strategies
    print("\n\nGenerated Defense Strategies:")
    for attack_type in ['sql_injection', 'xss', 'ddos']:
        strategy = trainer.generate_defense_strategy(attack_type)
        print(f"\n  {attack_type.upper()}:")
        print(f"    Detection rules: {len(strategy.get('detection_rules', []))}")
        print(f"    Mitigation steps: {len(strategy.get('mitigation_steps', []))}")
    
    system.shutdown()


# ============================================================================
# EXAMPLE 4: AUTO-UPGRADE SYSTEM
# ============================================================================

def example_auto_upgrade():
    """Example: System auto-upgrade and self-improvement"""
    print("\n" + "="*70)
    print("EXAMPLE 4: System Auto-Upgrade")
    print("="*70)
    
    from mlsystem.core.architecture import (
        MLSystemOrchestrator, ModuleConfig, ComponentType
    )
    from mlsystem.core.auto_upgrade import AutoUpgradeSystem
    
    # Create model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    system = MLSystemOrchestrator()
    
    # Auto-upgrade module
    upgrade_config = ModuleConfig(
        name='auto_upgrade',
        component_type=ComponentType.TRAINER,
        params={
            'model': model,
            'training_history': {
                'loss': [1.0, 0.8, 0.7, 0.7, 0.7]  # Converged/plateau
            }
        }
    )
    
    upgrader = AutoUpgradeSystem(upgrade_config)
    system.register_module(upgrader)
    
    # Analyze and upgrade
    print("\nAnalyzing system performance...")
    analysis = upgrader.analyze_performance()
    
    print(f"  Convergence Score: {analysis['overall_score']:.1f}")
    print(f"  Bottlenecks found: {len(analysis['bottlenecks'])}")
    print(f"  Optimization opportunities: {len(analysis['opportunities'])}")
    
    print("\nFetching improvements...")
    improvements = upgrader.fetch_improvements('llm')
    
    print(f"  Found {len(improvements)} improvements from external sources")
    
    print("\nApplying upgrades...")
    for improvement in improvements[:2]:
        if 'suggestions' in improvement:
            success = upgrader.apply_upgrade(improvement['suggestions'])
            print(f"  Applied: {improvement.get('source')} - {'✓' if success else '✗'}")
    
    upgrade_status = upgrader.get_upgrade_status()
    print(f"\n  Total upgrades applied: {upgrade_status['successful']}/{upgrade_status['attempted']}")
    
    system.shutdown()


# ============================================================================
# EXAMPLE 5: INTERACTIVE CHAT INTERFACE
# ============================================================================

def example_chat_interface():
    """Example: Interactive chat-based interface"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Interactive Chat Interface")
    print("="*70)
    
    from mlsystem.core.architecture import (
        MLSystemOrchestrator, ModuleConfig, ComponentType
    )
    from mlsystem.interface.chat import MLChatInterface
    
    # Create system with some modules
    system = MLSystemOrchestrator()
    
    # Add dummy modules
    print("\nInitializing system with modules...")
    
    # In real usage, you would:
    # system = MLSystemOrchestrator()
    # system.register_module(feeder)
    # system.register_module(encoder)
    # system.register_module(decoder)
    # system.register_module(reflector)
    # system.register_module(trainer)
    # system.set_pipeline(['feeder', 'encoder', 'decoder', 'reflector'])
    
    # Create chat interface
    chat = MLChatInterface(system)
    
    print("✓ System ready")
    print("\nYou can now interact with the system using commands like:")
    print("  - status          : Show system status")
    print("  - list_modules    : List all modules")
    print("  - run_inference   : Run inference")
    print("  - train           : Train the model")
    print("  - upgrade_system  : Trigger auto-upgrade")
    print("\nType 'help' for all commands")
    
    # Uncomment to start interactive mode:
    # chat.run()


# ============================================================================
# EXAMPLE 6: MULTI-DATA-TYPE PIPELINE
# ============================================================================

def example_multi_datatype():
    """Example: Pipeline handling multiple data types"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Multi-Data-Type Pipeline")
    print("="*70)
    
    from mlsystem.core.architecture import (
        MLSystemOrchestrator, ModuleConfig, ComponentType, DataType
    )
    from mlsystem.core.implementations import (
        ImageFeeder, TextFeeder, StatisticalFeeder
    )
    
    system = MLSystemOrchestrator()
    
    # Multiple input feeders
    feeders = [
        ('image_feeder', ImageFeeder),
        ('text_feeder', TextFeeder),
        ('stat_feeder', StatisticalFeeder)
    ]
    
    print("\nRegistering multi-modal feeders...")
    
    for name, FeederClass in feeders:
        config = ModuleConfig(
            name=name,
            component_type=ComponentType.FEEDER,
            input_types=[DataType.IMAGE] if 'image' in name else 
                       [DataType.TEXT] if 'text' in name else 
                       [DataType.STATISTICAL],
            enabled=True
        )
        
        feeder = FeederClass(config)
        system.register_module(feeder)
        print(f"  ✓ {name}")
    
    print(f"\nTotal feeders registered: {len(system.modules)}")
    
    system.shutdown()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "█"*70)
    print("█  HIERARCHICAL MAMBA + TRANSFORMER ML SYSTEM - EXAMPLES")
    print("█"*70)
    
    examples = [
        ("Basic Image Pipeline", example_basic_image_pipeline),
        ("Text with Reflector", example_text_with_reflector),
        ("Cybersecurity Training", example_cybersecurity_training),
        ("Auto-Upgrade System", example_auto_upgrade),
        ("Multi-Data-Type", example_multi_datatype),
        ("Chat Interface", example_chat_interface),
    ]
    
    for idx, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "█"*70)
    print("█  ALL EXAMPLES COMPLETED")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
