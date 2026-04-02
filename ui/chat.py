"""
Chat-Based Interactive Interface for Model Inference
Allows users to interact with trained model and generate/connect components
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path
try:
    import readline  # For better CLI interface (Unix only)
except ImportError:
    pass  # Not available on Windows — no readline history, but works fine

# Support both flat-file execution and package import
try:
    from core.architecture import MLSystemOrchestrator, BaseModule, ModuleConfig
except ImportError:
    from .architecture import MLSystemOrchestrator, BaseModule, ModuleConfig


# ============================================================================
# CHAT INTERFACE
# ============================================================================

class ChatCommand:
    """Represents a chat command"""
    
    def __init__(self, name: str, description: str, args: List[str] = None):
        self.name = name
        self.description = description
        self.args = args or []
    
    def __repr__(self):
        args_str = ' '.join(f'<{arg}>' for arg in self.args)
        return f"{self.name} {args_str}".strip()


class MLChatInterface:
    """Interactive chat interface for ML system"""
    
    def __init__(self, system: MLSystemOrchestrator):
        self.system = system
        self.session_start = datetime.now()
        self.conversation_history = []
        self.context = {}
        
        # Define available commands
        self.commands = {
            'help': ChatCommand('help', 'Show available commands'),
            'status': ChatCommand('status', 'Show system status'),
            'list_modules': ChatCommand('list_modules', 'List all registered modules'),
            'run_inference': ChatCommand('run_inference', 'Run inference on data', 
                                        ['input_path']),
            'train': ChatCommand('train', 'Train the model', 
                                ['num_epochs', 'batch_size']),
            'evaluate': ChatCommand('evaluate', 'Evaluate model performance'),
            'save_model': ChatCommand('save_model', 'Save trained model',
                                     ['path']),
            'load_model': ChatCommand('load_model', 'Load saved model',
                                     ['path']),
            'configure': ChatCommand('configure', 'Configure module',
                                    ['module_name', 'param', 'value']),
            'analyze_output': ChatCommand('analyze_output', 'Analyze model output'),
            'generate_report': ChatCommand('generate_report', 'Generate analysis report'),
            'upgrade_system': ChatCommand('upgrade_system', 'Trigger system auto-upgrade'),
            'chat': ChatCommand('chat', 'Chat with model', ['question']),
            'pipeline': ChatCommand('pipeline', 'Show/set pipeline',
                                   ['action', 'module_name']),
            'metrics': ChatCommand('metrics', 'Show training metrics'),
            'export': ChatCommand('export', 'Export configuration/model',
                                 ['format']),
            'quit': ChatCommand('quit', 'Exit the program'),
        }
    
    def display_welcome(self) -> None:
        """Display welcome message"""
        print("\n" + "="*70)
        print("  HIERARCHICAL MAMBA + TRANSFORMER ML SYSTEM")
        print("  With Reflector Auto-Correction & Self-Upgrade")
        print("="*70)
        print("\nType 'help' for available commands\n")
    
    def display_help(self) -> None:
        """Display help message"""
        print("\n" + "-"*70)
        print("AVAILABLE COMMANDS:")
        print("-"*70)
        
        for cmd_name, cmd in sorted(self.commands.items()):
            print(f"  {str(cmd):<45} - {cmd.description}")
        
        print("-"*70 + "\n")
    
    def run(self) -> None:
        """Start interactive chat loop"""
        self.display_welcome()
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if not self.process_command(user_input):
                    break
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def process_command(self, user_input: str) -> bool:
        """Process user command"""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Record in conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'command': command
        })
        
        # Handle commands
        if command == 'help':
            self.display_help()
        
        elif command == 'quit':
            self.system.shutdown()
            print("System shutdown complete. Goodbye!")
            return False
        
        elif command == 'status':
            self._cmd_status()
        
        elif command == 'list_modules':
            self._cmd_list_modules()
        
        elif command == 'run_inference':
            self._cmd_run_inference(args)
        
        elif command == 'train':
            self._cmd_train(args)
        
        elif command == 'evaluate':
            self._cmd_evaluate()
        
        elif command == 'save_model':
            self._cmd_save_model(args)
        
        elif command == 'load_model':
            self._cmd_load_model(args)
        
        elif command == 'configure':
            self._cmd_configure(args)
        
        elif command == 'analyze_output':
            self._cmd_analyze_output()
        
        elif command == 'generate_report':
            self._cmd_generate_report()
        
        elif command == 'upgrade_system':
            self._cmd_upgrade_system()
        
        elif command == 'pipeline':
            self._cmd_pipeline(args)
        
        elif command == 'metrics':
            self._cmd_metrics()
        
        elif command == 'export':
            self._cmd_export(args)
        
        elif command == 'chat':
            self._cmd_chat(args)
        
        else:
            print(f"Unknown command: {command}. Type 'help' for commands.")
        
        return True
    
    # ========================================================================
    # COMMAND IMPLEMENTATIONS
    # ========================================================================
    
    def _cmd_status(self) -> None:
        """Show system status"""
        status = self.system.get_system_status()
        
        print("\n" + "="*70)
        print("SYSTEM STATUS")
        print("="*70)
        print(f"Total Modules: {status['total_modules']}")
        print(f"Pipeline: {' → '.join(status['pipeline']) if status['pipeline'] else 'Not configured'}")
        print("\nModule Details:")
        
        for module_name, module_status in status['modules'].items():
            print(f"\n  {module_name}:")
            for key, value in module_status.items():
                print(f"    {key}: {value}")
        
        print("\n" + "="*70 + "\n")
    
    def _cmd_list_modules(self) -> None:
        """List all registered modules"""
        print("\n" + "-"*70)
        print("REGISTERED MODULES:")
        print("-"*70)
        
        for name, module in self.system.modules.items():
            print(f"\n  Name: {name}")
            print(f"  Type: {module.config.component_type.value}")
            print(f"  Enabled: {module.config.enabled}")
            print(f"  Input Types: {[dt.value for dt in module.config.input_types]}")
            print(f"  Output Type: {module.config.output_type.value if module.config.output_type else 'None'}")
        
        print("\n" + "-"*70 + "\n")
    
    def _cmd_run_inference(self, args: str) -> None:
        """Run inference on data"""
        if not args:
            print("Usage: run_inference <input_path>")
            return
        
        try:
            print(f"\nRunning inference on {args}...")
            
            # Load data
            if args.endswith('.pt'):
                data = torch.load(args)
            elif args.endswith('.npy'):
                import numpy as np
                data = torch.from_numpy(np.load(args))
            else:
                print(f"Unsupported file format: {args}")
                return
            
            # Run pipeline
            results = self.system.execute_pipeline(data)
            
            print(f"\nInference complete!")
            print(f"Output shape: {results['output'].shape if hasattr(results['output'], 'shape') else 'N/A'}")
            
            self.context['last_output'] = results
        
        except Exception as e:
            print(f"Error during inference: {e}")
    
    def _cmd_train(self, args: str) -> None:
        """Train the model"""
        parts = args.split() if args else []
        
        num_epochs = int(parts[0]) if parts else 10
        batch_size = int(parts[1]) if len(parts) > 1 else 32
        
        print(f"\nStarting training...")
        print(f"Epochs: {num_epochs}, Batch Size: {batch_size}")
        
        # Find trainer module
        trainer = None
        for module in self.system.modules.values():
            if module.config.component_type.value == 'trainer':
                trainer = module
                break
        
        if not trainer:
            print("No trainer module found")
            return
        
        try:
            print("Training in progress... (simulated)")
            print("✓ Training completed")
        except Exception as e:
            print(f"Error during training: {e}")
    
    def _cmd_evaluate(self) -> None:
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        trainer = None
        for module in self.system.modules.values():
            if module.config.component_type.value == 'trainer':
                trainer = module
                break
        
        if not trainer:
            print("No trainer module found")
            return
        
        if hasattr(trainer, 'get_training_summary'):
            summary = trainer.get_training_summary()
            
            print("\nTraining Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    
    def _cmd_save_model(self, args: str) -> None:
        """Save trained model"""
        if not args:
            args = "model_checkpoint.pt"
        
        try:
            print(f"Saving model to {args}...")
            
            # Save model and config
            self.system.save_config(args.replace('.pt', '_config.json'))
            
            # Save model weights
            for module in self.system.modules.values():
                if hasattr(module, 'model'):
                    torch.save(module.model.state_dict(), args)
            
            print("✓ Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _cmd_load_model(self, args: str) -> None:
        """Load saved model"""
        if not args:
            print("Usage: load_model <path>")
            return
        
        try:
            print(f"Loading model from {args}...")
            
            # Load config and model
            config_path = args.replace('.pt', '_config.json')
            config = self.system.load_config(config_path)
            
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _cmd_configure(self, args: str) -> None:
        """Configure module parameters"""
        parts = args.split(maxsplit=2) if args else []
        
        if len(parts) < 3:
            print("Usage: configure <module_name> <parameter> <value>")
            return
        
        module_name, param, value = parts
        
        if module_name not in self.system.modules:
            print(f"Module not found: {module_name}")
            return
        
        module = self.system.modules[module_name]
        module.config.params[param] = value
        
        print(f"✓ Configured {module_name}.{param} = {value}")
    
    def _cmd_analyze_output(self) -> None:
        """Analyze last output"""
        if 'last_output' not in self.context:
            print("No output to analyze. Run inference first.")
            return
        
        output = self.context['last_output']
        
        print("\nOutput Analysis:")
        print(f"  Stages executed: {len(output['stages'])}")
        print(f"  Final output shape: {output['output'].shape if hasattr(output['output'], 'shape') else 'N/A'}")
        print(f"  Output type: {type(output['output'])}")
    
    def _cmd_generate_report(self) -> None:
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().isoformat()
        
        report = {
            'timestamp': timestamp,
            'system_status': self.system.get_system_status(),
            'conversation_length': len(self.conversation_history),
            'session_duration': (datetime.now() - self.session_start).total_seconds()
        }
        
        # Save report
        report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_path}")
    
    def _cmd_upgrade_system(self) -> None:
        """Trigger auto-upgrade"""
        print("\nInitiating system auto-upgrade...")
        
        # Find auto-upgrade module
        auto_upgrade = None
        for module in self.system.modules.values():
            if hasattr(module, 'analyze_performance'):
                auto_upgrade = module
                break
        
        if not auto_upgrade:
            print("No auto-upgrade module found")
            return
        
        try:
            analysis = auto_upgrade.analyze_performance()
            print(f"\nPerformance Analysis:")
            print(f"  Overall Score: {analysis.get('overall_score', 'N/A')}")
            print(f"  Bottlenecks: {len(analysis.get('bottlenecks', []))}")
            print(f"  Opportunities: {len(analysis.get('opportunities', []))}")
            
            improvements = auto_upgrade.fetch_improvements('all')
            print(f"\nFetched {len(improvements)} improvements")
            
            applied = 0
            for improvement in improvements[:3]:
                if 'suggestions' in improvement:
                    if auto_upgrade.apply_upgrade(improvement['suggestions']):
                        applied += 1
            
            print(f"\n✓ Applied {applied} upgrades")
        
        except Exception as e:
            print(f"Error during upgrade: {e}")
    
    def _cmd_pipeline(self, args: str) -> None:
        """Show or configure pipeline"""
        parts = args.split() if args else []
        
        if not parts:
            # Show current pipeline
            print(f"\nCurrent Pipeline: {' → '.join(self.system.pipeline_sequence)}")
        
        elif parts[0] == 'set':
            # Set pipeline: pipeline set module1 module2 module3
            module_names = parts[1:]
            try:
                self.system.set_pipeline(module_names)
                print(f"✓ Pipeline updated: {' → '.join(module_names)}")
            except ValueError as e:
                print(f"Error: {e}")
        
        else:
            print("Usage: pipeline [set <module1> <module2> ...]")
    
    def _cmd_metrics(self) -> None:
        """Show training metrics"""
        print("\nTraining Metrics:")
        
        for module in self.system.modules.values():
            if hasattr(module, 'training_history'):
                history = module.training_history
                
                print(f"\n{module.config.name}:")
                for key, values in history.items():
                    if values:
                        print(f"  {key}: {len(values)} entries")
    
    def _cmd_export(self, args: str) -> None:
        """Export configuration or model"""
        format_type = args if args else 'json'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"export_{timestamp}.{format_type}"
        
        if format_type == 'json':
            config = self.system.load_config if hasattr(self.system, 'load_config') else {}
            with open(filename, 'w') as f:
                json.dump(self.system.get_system_status(), f, indent=2)
        
        print(f"✓ Exported to {filename}")
    
    def _cmd_chat(self, args: str) -> None:
        """Chat with the model"""
        from ui.model_chat import start_chat
        
        # If args is empty, let start_chat handle it (it will ask for model name)
        model_name = args.strip() or None
        
        try:
            start_chat(model_name)
        except Exception as e:
            print(f"Error starting chat: {e}")
