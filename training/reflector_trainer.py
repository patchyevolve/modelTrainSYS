"""
Reflector module for auto-correction and integrated trainer.
NeuralReflector: pure PyTorch confidence scoring + blend correction.
LLMReflector: uses Groq llama-3.3-70b-versatile for text output validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import json
import logging
from dataclasses import dataclass

# Support both flat-file execution and package import
try:
    from core.architecture import Reflector, Trainer, ModuleConfig, ComponentType
except ImportError:
    from .architecture import Reflector, Trainer, ModuleConfig, ComponentType

log = logging.getLogger("Reflector")

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"


def _groq_chat(messages: List[Dict], temperature: float = 0.2,
               max_tokens: int = 512) -> str:
    """Minimal synchronous Groq call. Returns assistant content."""
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": GROQ_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        GROQ_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {GROQ_API_KEY}",
                 "User-Agent": "python-groq-client/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"Groq call failed: {e}")
        return ""


# ============================================================================
# REFLECTOR MODULE
# ============================================================================

@dataclass
class ReflectionResult:
    """Container for reflection results"""
    original_output: Any
    corrected_output: Any
    confidence: float
    corrections_made: List[str]
    quality_score: float
    metadata: Dict[str, Any]


class NeuralReflector(Reflector):
    """
    Neural network-based reflector for validation and auto-correction.
    Uses a secondary network to validate and refine outputs.
    """
    
    def initialize(self) -> None:
        self.input_dim = self.config.params.get('input_dim', 512)
        self.hidden_dim = self.config.params.get('hidden_dim', 256)
        self.num_correction_layers = self.config.params.get('num_correction_layers', 2)
        
        # Validation head
        self.validator = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Correction head (refines output)
        self.corrector = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(self.hidden_dim)
                )
                for _ in range(self.num_correction_layers)
            ],
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        # Loss tracking
        self.validation_threshold = self.config.params.get('threshold', 0.7)
        
        super().initialize()
    
    def get_confidence_score(self, output: torch.Tensor) -> float:
        """Get confidence in output correctness"""
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output).float()
        
        with torch.no_grad():
            # Flatten if needed
            if output.dim() > 2:
                output = output.view(output.size(0), -1)
            
            confidence = self.validator(output)
        
        return confidence.mean().item()
    
    def reflect(self, output: torch.Tensor, 
                ground_truth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Validate output and suggest corrections
        """
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output).float()
        
        original_output = output.clone().detach()
        
        # Get confidence score
        confidence = self.get_confidence_score(output)
        
        # Apply corrections if confidence is low
        corrections_made = []
        if confidence < self.validation_threshold:
            if isinstance(output, np.ndarray):
                output = torch.from_numpy(output).float()
            
            if output.dim() > 2:
                output_flat = output.view(output.size(0), -1)
            else:
                output_flat = output
            
            corrected = self.corrector(output_flat)
            
            # Blend original and corrected based on confidence
            blend_factor = 1.0 - confidence
            output = original_output + blend_factor * (corrected - original_output)
            corrections_made.append(f"Applied blend-correction (factor: {blend_factor:.2f})")
        
        # Calculate quality score
        if ground_truth is not None:
            if isinstance(ground_truth, np.ndarray):
                ground_truth = torch.from_numpy(ground_truth).float()
            
            mse = torch.mean((output - ground_truth) ** 2).item()
            quality_score = 1.0 / (1.0 + mse)
        else:
            quality_score = confidence
        
        metadata = {
            'confidence': float(confidence),
            'quality_score': float(quality_score),
            'corrections_made': corrections_made,
            'threshold_exceeded': confidence < self.validation_threshold
        }
        
        return output, metadata
    
    def forward(self, data: Any) -> Any:
        """Process through reflector"""
        corrected, metadata = self.reflect(data)
        return corrected


class EnsembleReflector(Reflector):
    """
    Ensemble of multiple reflectors for robust validation
    """
    
    def initialize(self) -> None:
        self.num_reflectors = self.config.params.get('num_reflectors', 3)
        self.voting_strategy = self.config.params.get('voting', 'majority')
        
        self.reflectors = nn.ModuleList([
            NeuralReflector(self.config) 
            for _ in range(self.num_reflectors)
        ])
        
        for reflector in self.reflectors:
            reflector.initialize()
        
        super().initialize()
    
    def get_confidence_score(self, output: torch.Tensor) -> float:
        """Average confidence across all reflectors"""
        scores = [r.get_confidence_score(output) for r in self.reflectors]
        return np.mean(scores)
    
    def reflect(self, output: torch.Tensor, 
                ground_truth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Ensemble reflection"""
        results = []
        corrections_list = []
        confidences = []
        
        for reflector in self.reflectors:
            corrected, metadata = reflector.reflect(output, ground_truth)
            results.append(corrected)
            confidences.append(metadata['confidence'])
            corrections_list.extend(metadata.get('corrections_made', []))
        
        # Average corrections
        if self.voting_strategy == 'majority':
            final_output = torch.stack(results).mean(dim=0)
        else:  # weighted
            weights = torch.softmax(torch.tensor(confidences, dtype=torch.float32), dim=0)
            final_output = sum(r * w for r, w in zip(results, weights))
        
        metadata = {
            'confidence': float(np.mean(confidences)),
            'reflector_confidences': confidences,
            'corrections_made': list(set(corrections_list)),
            'ensemble_size': self.num_reflectors
        }
        
        return final_output, metadata
    
    def forward(self, data: Any) -> Any:
        """Process through ensemble reflector"""
        corrected, _ = self.reflect(data)
        return corrected


# ============================================================================
# LLM REFLECTOR (Groq-backed, for text outputs)
# ============================================================================

class LLMReflector(Reflector):
    """
    Uses Groq llama-3.3-70b-versatile to validate and correct text outputs.
    Falls back to NeuralReflector confidence scoring for non-text tensors.
    """

    def initialize(self) -> None:
        self.input_dim = self.config.params.get("input_dim", 512)
        # Neural fallback for tensor confidence
        self._neural = NeuralReflector(ModuleConfig(
            name=self.config.name + "_neural",
            component_type=self.config.component_type,
            params={"input_dim": self.input_dim,
                    "hidden_dim": self.config.params.get("hidden_dim", 256)},
        ))
        self._neural.initialize()
        self.validation_threshold = self.config.params.get("threshold", 0.7)
        super().initialize()

    def get_confidence_score(self, output: Any) -> float:
        """For tensors use neural validator; for strings use LLM."""
        if isinstance(output, (torch.Tensor, np.ndarray)):
            if isinstance(output, np.ndarray):
                output = torch.from_numpy(output).float()
            return self._neural.get_confidence_score(output)
        # String output — ask LLM
        if isinstance(output, str) and output.strip():
            return self._llm_confidence(output)
        return 0.5

    def _llm_confidence(self, text: str) -> float:
        """Ask Groq to rate output quality 0-1."""
        messages = [
            {"role": "system",
             "content": ("You are a quality evaluator. "
                         "Rate the following ML model output on a scale 0.0 to 1.0 "
                         "for coherence, correctness, and completeness. "
                         "Reply with ONLY a float number, nothing else.")},
            {"role": "user", "content": text[:800]},
        ]
        raw = _groq_chat(messages, temperature=0.0, max_tokens=10)
        try:
            score = float(raw.strip().split()[0])
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def reflect(self, output: Any,
                ground_truth: Optional[Any] = None) -> Tuple[Any, Dict]:
        """
        For tensors: neural blend-correction.
        For strings: LLM rewrite if confidence is low.
        """
        if isinstance(output, (torch.Tensor, np.ndarray)):
            return self._neural.reflect(output, ground_truth)

        if isinstance(output, str):
            confidence = self._llm_confidence(output)
            corrections_made = []
            corrected = output

            if confidence < self.validation_threshold:
                corrected = self._llm_correct(output, ground_truth)
                corrections_made.append(f"LLM rewrite (confidence={confidence:.2f})")

            quality = confidence if ground_truth is None else (
                self._llm_confidence(corrected)
            )
            return corrected, {
                "confidence":    confidence,
                "quality_score": quality,
                "corrections_made": corrections_made,
                "threshold_exceeded": confidence < self.validation_threshold,
            }

        # Unknown type — pass through
        return output, {"confidence": 0.5, "quality_score": 0.5,
                        "corrections_made": [], "threshold_exceeded": False}

    def _llm_correct(self, text: str, ground_truth: Any = None) -> str:
        """Ask Groq to improve the output."""
        gt_hint = f"\nExpected context: {str(ground_truth)[:200]}" if ground_truth else ""
        messages = [
            {"role": "system",
             "content": ("You are an ML output corrector. "
                         "Improve the following model output for clarity and correctness. "
                         "Return ONLY the improved text, no explanation.")},
            {"role": "user",
             "content": f"Original output:\n{text[:800]}{gt_hint}"},
        ]
        corrected = _groq_chat(messages, temperature=0.3, max_tokens=400)
        return corrected if corrected.strip() else text

    def forward(self, data: Any) -> Any:
        corrected, _ = self.reflect(data)
        return corrected


# ============================================================================
# TRAINER WITH REFLECTOR INTEGRATION
# ============================================================================

class ReflectorIntegratedTrainer(Trainer):
    """
    Trainer that integrates reflector feedback for improved training
    and faster convergence
    """
    
    def initialize(self) -> None:
        self.model = self.config.params.get('model')
        self.reflector = self.config.params.get('reflector')
        self.optimizer_type = self.config.params.get('optimizer', 'adam')
        self.learning_rate = self.config.params.get('lr', 1e-3)
        self.reflector_weight = self.config.params.get('reflector_weight', 0.3)
        
        if not self.model:
            raise ValueError("Model required in params")
        if not self.reflector:
            raise ValueError("Reflector required in params")
        
        # Setup optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate
            )
        
        self.loss_fn = nn.MSELoss()
        self.training_history = {
            'loss': [],
            'reflector_loss': [],
            'total_loss': [],
            'val_loss': []
        }
        
        super().initialize()
    
    def get_reflector_loss(self, output: torch.Tensor, 
                          target: torch.Tensor) -> float:
        """
        Compute loss based on reflector feedback.
        Lower confidence = higher loss to encourage correction
        """
        corrected, metadata = self.reflector.reflect(output, target)
        
        # Loss components
        reconstruction_loss = self.loss_fn(corrected, target)
        confidence = torch.tensor(metadata['confidence'])
        
        # Reflector loss encourages high confidence
        reflector_loss = -torch.log(confidence + 1e-8)
        
        return reflector_loss.item()
    
    def train_step(self, batch: torch.Tensor, 
                  labels: torch.Tensor) -> Dict[str, float]:
        """Single training step with reflector feedback"""
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(batch)
        
        # Primary loss
        primary_loss = self.loss_fn(output, labels)
        
        # Reflector loss
        reflector_loss = torch.tensor(
            self.get_reflector_loss(output, labels)
        )
        
        # Combined loss
        total_loss = primary_loss + self.reflector_weight * reflector_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Record metrics
        metrics = {
            'primary_loss': primary_loss.item(),
            'reflector_loss': reflector_loss.item(),
            'total_loss': total_loss.item(),
            'batch_size': batch.shape[0]
        }
        
        self.training_history['loss'].append(metrics['primary_loss'])
        self.training_history['reflector_loss'].append(metrics['reflector_loss'])
        self.training_history['total_loss'].append(metrics['total_loss'])
        
        return metrics
    
    def validate(self, val_data: torch.Tensor, 
                val_labels: torch.Tensor) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(val_data)
            val_loss = self.loss_fn(output, val_labels)
            
            # Reflector confidence
            confidence = self.reflector.get_confidence_score(output)
        
        self.training_history['val_loss'].append(val_loss.item())
        
        metrics = {
            'val_loss': val_loss.item(),
            'val_confidence': confidence
        }
        
        self.model.train()
        
        return metrics
    
    def train_epoch(self, train_loader, val_loader=None, 
                   num_epochs: int = 1) -> Dict[str, List]:
        """Train for multiple epochs"""
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in train_loader:
                metrics = self.train_step(batch_data, batch_labels)
                epoch_loss += metrics['total_loss']
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            if val_loader:
                val_metrics = self.validate(*next(iter(val_loader)))
                self.logger.info(
                    f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                    f"Confidence: {val_metrics['val_confidence']:.4f}"
                )
        
        return self.training_history
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Process data through trainer (inference mode)"""
        self.model.eval()
        with torch.no_grad():
            return self.model(data)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'total_steps': len(self.training_history['loss']),
            'avg_loss': float(np.mean(self.training_history['loss'])),
            'final_loss': float(self.training_history['loss'][-1]) 
                         if self.training_history['loss'] else None,
            'avg_reflector_loss': float(np.mean(self.training_history['reflector_loss']))
                                 if self.training_history['reflector_loss'] else None,
            'avg_val_loss': float(np.mean(self.training_history['val_loss']))
                           if self.training_history['val_loss'] else None
        }
