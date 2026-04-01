# ML SYSTEM - QUICK REFERENCE GUIDE

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Module Types](#module-types)
4. [Data Types](#data-types)
5. [Common Patterns](#common-patterns)
6. [Commands Reference](#commands-reference)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Install from source
git clone <repo>
cd mlsystem
pip install -e .

# Install dependencies
pip install torch numpy pillow

# Optional: Build C++ extensions for performance
python setup.py build_ext --inplace
```

---

## Basic Usage

### 1. Create System

```python
from mlsystem import MLSystemOrchestrator

system = MLSystemOrchestrator()
```

### 2. Add Modules

```python
from mlsystem import (
    ModuleConfig, ComponentType, DataType,
    ImageFeeder, HierarchicalMambaEncoder, TransformerDecoder
)

# Image input
feeder = ImageFeeder(ModuleConfig(
    name='feeder',
    component_type=ComponentType.FEEDER,
    input_types=[DataType.IMAGE]
))
system.register_module(feeder)

# Encoder
encoder = HierarchicalMambaEncoder(ModuleConfig(
    name='encoder',
    component_type=ComponentType.ENCODER,
    params={'hidden_dim': 512, 'num_layers': 3}
))
system.register_module(encoder)

# Decoder
decoder = TransformerDecoder(ModuleConfig(
    name='decoder',
    component_type=ComponentType.DECODER,
    params={'latent_dim': 512, 'output_dim': 256}
))
system.register_module(decoder)
```

### 3. Configure Pipeline

```python
system.set_pipeline(['feeder', 'encoder', 'decoder'])
```

### 4. Run Inference

```python
results = system.execute_pipeline(input_data)
output = results['output']
```

### 5. Shutdown

```python
system.shutdown()
```

---

## Module Types

### FEEDER Modules

| Type | Usage | Input | Output |
|------|-------|-------|--------|
| `ImageFeeder` | Process images | Image path/array | Tensor |
| `TextFeeder` | Process text | String/list | Token tensor |
| `StatisticalFeeder` | Process numerical | Array/tensor | Normalized tensor |

**Create Custom:**
```python
from mlsystem import DataFeeder, ModuleConfig

class CustomFeeder(DataFeeder):
    def initialize(self): pass
    def validate_data(self, data): return True
    def preprocess(self, data): return processed
    def load_batch(self, batch_size, **kwargs): return batch, metadata
    def forward(self, data): return self.preprocess(data)
```

### ENCODER Modules

| Type | Features |
|------|----------|
| `HierarchicalMambaEncoder` | Multi-scale state-space model, residual connections |

**Parameters:**
- `input_dim`: Input dimension (default: 256)
- `hidden_dim`: Hidden dimension (default: 512)
- `num_layers`: Number of layers (default: 3)
- `num_scales`: Number of scales (default: 3)

### DECODER Modules

| Type | Features |
|------|----------|
| `TransformerDecoder` | Standard transformer decoder, multi-head attention |

**Parameters:**
- `latent_dim`: Latent dimension (default: 512)
- `output_dim`: Output dimension (default: 256)
- `num_heads`: Attention heads (default: 8)
- `num_layers`: Decoder layers (default: 3)

### REFLECTOR Modules

| Type | Purpose |
|------|---------|
| `NeuralReflector` | Single reflector for output validation |
| `EnsembleReflector` | Multiple reflectors for robustness |

**Methods:**
- `reflect(output, ground_truth)`: Validate and correct output
- `get_confidence_score(output)`: Confidence 0-1

### TRAINER Modules

| Type | Features |
|------|----------|
| `ReflectorIntegratedTrainer` | Training with reflector feedback |
| `CybersecurityTrainer` | Attack detection training |

**Methods:**
- `train_step(batch, labels)`: Single training iteration
- `validate(val_data, val_labels)`: Validation
- `train_epoch(loader, val_loader, num_epochs)`: Full epoch

---

## Data Types

```python
from mlsystem import DataType

# Supported types:
DataType.IMAGE           # Images (PNG, JPG, BMP, etc.)
DataType.TEXT            # Text and NLP data
DataType.STATISTICAL     # Numerical/tabular data
DataType.AUDIO           # Audio signals
DataType.VIDEO           # Video streams
DataType.CUSTOM          # Custom format
```

---

## Common Patterns

### Pattern 1: Image Classification

```python
system = MLSystemOrchestrator()

# Setup
feeder = ImageFeeder(ModuleConfig('feeder', ComponentType.FEEDER))
encoder = HierarchicalMambaEncoder(ModuleConfig('encoder', ComponentType.ENCODER))
decoder = TransformerDecoder(ModuleConfig('decoder', ComponentType.DECODER,
                                         params={'output_dim': 1000}))
reflector = NeuralReflector(ModuleConfig('reflector', ComponentType.REFLECTOR))

system.register_module(feeder)
system.register_module(encoder)
system.register_module(decoder)
system.register_module(reflector)

system.set_pipeline(['feeder', 'encoder', 'decoder', 'reflector'])

# Inference
image_path = 'path/to/image.jpg'
results = system.execute_pipeline(image_path)
predictions = results['output']
```

### Pattern 2: Text Generation with Correction

```python
system = MLSystemOrchestrator()

# Add modules
system.register_module(TextFeeder(ModuleConfig('text_feeder', ComponentType.FEEDER)))
system.register_module(HierarchicalMambaEncoder(ModuleConfig('encoder', ComponentType.ENCODER)))
system.register_module(TransformerDecoder(ModuleConfig('decoder', ComponentType.DECODER)))
system.register_module(NeuralReflector(ModuleConfig('reflector', ComponentType.REFLECTOR)))

system.set_pipeline(['text_feeder', 'encoder', 'decoder', 'reflector'])

# Use
text = "Generate text from this prompt"
results = system.execute_pipeline(text)
corrected_output = results['stages']['reflector']
```

### Pattern 3: Multi-Modal Processing

```python
system = MLSystemOrchestrator()

# Add multiple feeders
system.register_module(ImageFeeder(ModuleConfig('img', ComponentType.FEEDER)))
system.register_module(TextFeeder(ModuleConfig('txt', ComponentType.FEEDER)))
system.register_module(StatisticalFeeder(ModuleConfig('num', ComponentType.FEEDER)))

# Process different data types
image_results = system.execute_pipeline(image_path)
text_results = system.execute_pipeline(text_data)
statistical_results = system.execute_pipeline(numerical_data)
```

### Pattern 4: Training with Auto-Correction

```python
model = nn.Sequential(...)
reflector = NeuralReflector(ModuleConfig('reflector', ComponentType.REFLECTOR))

trainer = ReflectorIntegratedTrainer(ModuleConfig(
    'trainer', ComponentType.TRAINER,
    params={
        'model': model,
        'reflector': reflector,
        'lr': 1e-3,
        'reflector_weight': 0.3
    }
))

# Train
for epoch in range(10):
    metrics = trainer.train_step(batch_x, batch_y)
    print(f"Loss: {metrics['total_loss']:.4f}")
```

### Pattern 5: Cybersecurity Training

```python
from mlsystem.cybersec import CybersecurityTrainer, AttackPatternGenerator

trainer = CybersecurityTrainer(config)
gen = AttackPatternGenerator()

for epoch in range(10):
    # Generate attacks
    attacks = gen.generate_attack_batch(32)
    attack_features, attack_labels = trainer.generate_training_data(32)
    
    # Generate benign
    benign_features, benign_labels = trainer.generate_benign_data(32)
    
    # Train
    all_x = torch.cat([attack_features, benign_features])
    all_y = torch.cat([attack_labels, benign_labels])
    metrics = trainer.train_step_cybersec(all_x, all_y)
```

### Pattern 6: Auto-Upgrade

```python
from mlsystem.core.auto_upgrade import AutoUpgradeSystem

upgrader = AutoUpgradeSystem(ModuleConfig(
    'upgrader', ComponentType.TRAINER,
    params={'model': model, 'training_history': history}
))

# Analyze
analysis = upgrader.analyze_performance()

# Fetch improvements
improvements = upgrader.fetch_improvements('all')

# Apply
for improvement in improvements[:3]:
    upgrader.apply_upgrade(improvement)
```

---

## Commands Reference

### Interactive Chat Mode

```bash
# Start chat
python -c "from mlsystem import MLChatInterface; chat = MLChatInterface(system); chat.run()"

# Available commands:
>>> help                      # Show all commands
>>> status                     # System status
>>> list_modules              # Show modules
>>> run_inference path.jpg    # Inference
>>> train 50 32              # Train 50 epochs, batch 32
>>> evaluate                  # Evaluate performance
>>> configure module param val # Set parameter
>>> pipeline set m1 m2 m3    # Set pipeline
>>> upgrade_system            # Auto-upgrade
>>> chat "Your question"      # Chat with model
>>> save_model checkpoint.pt  # Save model
>>> load_model checkpoint.pt  # Load model
>>> metrics                   # Show metrics
>>> generate_report           # Generate report
>>> quit                      # Exit
```

---

## Performance Tips

### 1. GPU Acceleration

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move modules to GPU
for module in system.modules.values():
    if hasattr(module, 'to'):
        module.to(device)
```

### 2. Batch Processing

```python
# Optimal batch size based on memory
batch_size = 64  # Adjust for your GPU

# Or use gradient accumulation
accumulation_steps = 4
```

### 3. Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = system.execute_pipeline(data)
```

### 4. Multi-Threading

```python
# Parallel execution
results = system.execute_pipeline(data, parallel=True)
```

### 5. Model Optimization

```python
# C++ kernels (if built)
import mlsystem.cpp as cpp

# Use fast kernels for critical operations
encoded = cpp.mamba_forward(x, A, B, C, D)
```

### 6. Checkpointing

```python
# Save during training
system.save_config('checkpoint.json')
torch.save(model.state_dict(), 'model.pt')

# Resume
config = system.load_config('checkpoint.json')
model.load_state_dict(torch.load('model.pt'))
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution:**
```python
# Reduce batch size
config.params['batch_size'] = 16

# Reduce model size
config.params['hidden_dim'] = 256
config.params['num_layers'] = 2

# Use gradient checkpointing
torch.utils.checkpoint.checkpoint(module, x)
```

### Issue: Slow Training

**Solution:**
```python
# Enable parallel execution
system.execute_pipeline(data, parallel=True)

# Use C++ extensions
# python setup.py build_ext --inplace

# Reduce model complexity
config.params['num_scales'] = 2
```

### Issue: Poor Accuracy

**Solution:**
```python
# Increase reflector weight
config.params['reflector_weight'] = 0.5

# Add regularization
config.params['dropout'] = 0.3

# Trigger auto-upgrade
upgrader.fetch_improvements('all')
upgrader.apply_upgrade(...)
```

### Issue: Module Not Found

**Solution:**
```python
# Check registration
print(system.modules.keys())

# Check spelling
print(list(system.modules.keys()))

# Register missing module
system.register_module(module)
```

### Issue: Pipeline Execution Error

**Solution:**
```python
# Verify pipeline
print(system.pipeline_sequence)

# Check module compatibility
for module_name in system.pipeline_sequence:
    module = system.modules[module_name]
    print(f"{module_name}: {module.config.enabled}")

# Verify data flow
results = system.execute_pipeline(data, parallel=False)
for stage, output in results['stages'].items():
    print(f"{stage}: {output.shape if hasattr(output, 'shape') else type(output)}")
```

---

## Advanced Configuration

### Custom Loss Function

```python
def custom_loss(output, target, reflector, alpha=0.3, beta=0.2):
    # Primary loss
    mse_loss = nn.MSELoss()(output, target)
    
    # Reflector loss
    _, metadata = reflector.reflect(output, target)
    confidence = metadata['confidence']
    reflector_loss = -torch.log(torch.tensor(confidence) + 1e-8)
    
    # Adversarial loss
    # ... (custom implementation)
    
    total = (1-alpha-beta) * mse_loss + alpha * reflector_loss
    return total
```

### Custom Metrics

```python
def compute_metrics(predictions, targets, reflector):
    return {
        'accuracy': (predictions.argmax(1) == targets).float().mean(),
        'confidence': reflector.get_confidence_score(predictions),
        'f1': compute_f1(predictions, targets),
        'loss': loss_fn(predictions, targets)
    }
```

### Custom Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

for epoch in range(100):
    loss = train_epoch()
    scheduler.step(loss)
```

---

## API Cheat Sheet

```python
# Create system
system = MLSystemOrchestrator()

# Register module
system.register_module(module)

# Set pipeline
system.set_pipeline(['module1', 'module2', ...])

# Execute
results = system.execute_pipeline(data, parallel=False)

# Configuration
system.save_config('config.json')
config = system.load_config('config.json')

# Status
status = system.get_system_status()

# Shutdown
system.shutdown()

# Reflector
corrected, metadata = reflector.reflect(output, ground_truth)
confidence = reflector.get_confidence_score(output)

# Trainer
metrics = trainer.train_step(batch_x, batch_y)
summary = trainer.get_training_summary()
history = trainer.train_epoch(train_loader, val_loader, num_epochs)

# Auto-Upgrade
analysis = upgrader.analyze_performance()
improvements = upgrader.fetch_improvements('all')
upgrader.apply_upgrade(improvement)

# Chat
chat = MLChatInterface(system)
chat.run()
```

---

## Resources

- **Documentation**: See `README.md` for detailed docs
- **Examples**: Check `examples/integration_examples.py` for usage patterns
- **GitHub**: Report issues and contribute improvements
- **Citation**: See README for citation information

---

**Version**: 1.0  
**Last Updated**: 2024
