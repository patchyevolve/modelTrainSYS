"""
Cybersecurity-specific training module with offense generation
Handles real-time and synthetic attack patterns
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json

# Support both flat-file execution and package import
try:
    from .architecture import Trainer, ModuleConfig, ComponentType
    from .reflector_trainer import ReflectorIntegratedTrainer
except ImportError:
    from architecture import Trainer, ModuleConfig, ComponentType
    from reflector_trainer import ReflectorIntegratedTrainer


# ============================================================================
# ATTACK PATTERN GENERATOR
# ============================================================================

class AttackPatternGenerator:
    """
    Generates realistic attack patterns for training.
    Supports both real-time feeds and synthetic generation.
    """
    
    def __init__(self):
        self.attack_types = {
            'sql_injection': self._generate_sql_injection,
            'xss': self._generate_xss,
            'buffer_overflow': self._generate_buffer_overflow,
            'ddos': self._generate_ddos,
            'malware': self._generate_malware,
            'privilege_escalation': self._generate_privilege_escalation,
            'credential_stuffing': self._generate_credential_stuffing,
            'zero_day': self._generate_zero_day,
        }
        
        self.real_time_feeds = []
        self.synthetic_pool = []
    
    def _generate_sql_injection(self) -> Dict[str, Any]:
        """Generate SQL injection pattern"""
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users;--",
            "' UNION SELECT * FROM passwords--",
            "1' AND '1'='1",
            "admin'--",
        ]
        
        return {
            'type': 'sql_injection',
            'payload': np.random.choice(payloads),
            'severity': 'high',
            'timestamp': datetime.now().isoformat(),
            'vector': 'query_parameter',
            'detection_features': self._extract_sql_features()
        }
    
    def _generate_xss(self) -> Dict[str, Any]:
        """Generate XSS pattern"""
        payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(1)'></iframe>",
        ]
        
        return {
            'type': 'xss',
            'payload': np.random.choice(payloads),
            'severity': 'high',
            'timestamp': datetime.now().isoformat(),
            'vector': 'user_input',
            'detection_features': self._extract_xss_features()
        }
    
    def _generate_buffer_overflow(self) -> Dict[str, Any]:
        """Generate buffer overflow pattern"""
        overflow_sizes = [256, 512, 1024, 2048, 4096]
        
        return {
            'type': 'buffer_overflow',
            'payload_size': np.random.choice(overflow_sizes),
            'offset': np.random.randint(100, 500),
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'vector': 'memory_corruption',
            'detection_features': self._extract_buffer_features()
        }
    
    def _generate_ddos(self) -> Dict[str, Any]:
        """Generate DDoS pattern"""
        return {
            'type': 'ddos',
            'requests_per_second': np.random.randint(1000, 100000),
            'source_ips': np.random.randint(10, 1000),
            'pattern': np.random.choice(['flood', 'slowloris', 'amplification']),
            'severity': 'high',
            'timestamp': datetime.now().isoformat(),
            'vector': 'network_flooding',
            'detection_features': self._extract_ddos_features()
        }
    
    def _generate_malware(self) -> Dict[str, Any]:
        """Generate malware signature pattern"""
        return {
            'type': 'malware',
            'family': np.random.choice(['ransomware', 'trojan', 'worm', 'botnet']),
            'obfuscation_level': np.random.randint(0, 10),
            'evasion_techniques': np.random.randint(0, 5),
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'vector': 'executable_code',
            'detection_features': self._extract_malware_features()
        }
    
    def _generate_privilege_escalation(self) -> Dict[str, Any]:
        """Generate privilege escalation pattern"""
        return {
            'type': 'privilege_escalation',
            'escalation_path': np.random.choice(['kernel', 'syscall', 'service']),
            'user_level': np.random.choice(['user', 'system', 'admin']),
            'target_level': 'root',
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'vector': 'permission_bypass',
            'detection_features': self._extract_priv_esc_features()
        }
    
    def _generate_credential_stuffing(self) -> Dict[str, Any]:
        """Generate credential stuffing pattern"""
        return {
            'type': 'credential_stuffing',
            'attempts': np.random.randint(100, 10000),
            'unique_credentials': np.random.randint(50, 1000),
            'success_rate': np.random.uniform(0, 0.2),
            'severity': 'medium',
            'timestamp': datetime.now().isoformat(),
            'vector': 'authentication',
            'detection_features': self._extract_credential_features()
        }
    
    def _generate_zero_day(self) -> Dict[str, Any]:
        """Generate zero-day pattern"""
        return {
            'type': 'zero_day',
            'component': np.random.choice(['kernel', 'browser', 'app', 'lib']),
            'cve_similarity': np.random.uniform(0, 1),
            'exploit_reliability': np.random.uniform(0, 1),
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'vector': 'unknown',
            'detection_features': self._extract_zero_day_features()
        }
    
    @staticmethod
    def _extract_sql_features() -> List[float]:
        """Extract features for SQL injection"""
        return [
            np.random.uniform(0, 1),  # keyword_density
            np.random.uniform(0, 1),  # special_char_ratio
            np.random.uniform(0, 1),  # comment_count
            np.random.uniform(0, 1),  # union_presence
        ]
    
    @staticmethod
    def _extract_xss_features() -> List[float]:
        """Extract features for XSS"""
        return [
            np.random.uniform(0, 1),  # script_tag_density
            np.random.uniform(0, 1),  # event_handler_ratio
            np.random.uniform(0, 1),  # encoding_obfuscation
            np.random.uniform(0, 1),  # external_resource_count
        ]
    
    @staticmethod
    def _extract_buffer_features() -> List[float]:
        """Extract features for buffer overflow"""
        return [
            np.random.uniform(0, 1),  # overflow_size_ratio
            np.random.uniform(0, 1),  # nop_sled_density
            np.random.uniform(0, 1),  # shellcode_entropy
            np.random.uniform(0, 1),  # address_alignment
        ]
    
    @staticmethod
    def _extract_ddos_features() -> List[float]:
        """Extract features for DDoS"""
        return [
            np.random.uniform(0, 1),  # request_rate
            np.random.uniform(0, 1),  # source_diversity
            np.random.uniform(0, 1),  # packet_similarity
            np.random.uniform(0, 1),  # geographic_spread
        ]
    
    @staticmethod
    def _extract_malware_features() -> List[float]:
        """Extract features for malware"""
        return [
            np.random.uniform(0, 1),  # entropy
            np.random.uniform(0, 1),  # api_call_count
            np.random.uniform(0, 1),  # string_obfuscation
            np.random.uniform(0, 1),  # packed_score
        ]
    
    @staticmethod
    def _extract_priv_esc_features() -> List[float]:
        """Extract features for privilege escalation"""
        return [
            np.random.uniform(0, 1),  # syscall_anomaly
            np.random.uniform(0, 1),  # permission_change_rate
            np.random.uniform(0, 1),  # capability_bypass
            np.random.uniform(0, 1),  # execution_context_change
        ]
    
    @staticmethod
    def _extract_credential_features() -> List[float]:
        """Extract features for credential stuffing"""
        return [
            np.random.uniform(0, 1),  # attempt_rate
            np.random.uniform(0, 1),  # credential_diversity
            np.random.uniform(0, 1),  # failure_pattern
            np.random.uniform(0, 1),  # source_concentration
        ]
    
    @staticmethod
    def _extract_zero_day_features() -> List[float]:
        """Extract features for zero-day"""
        return [
            np.random.uniform(0, 1),  # cve_similarity_score
            np.random.uniform(0, 1),  # exploitation_technique_novelty
            np.random.uniform(0, 1),  # bypass_depth
            np.random.uniform(0, 1),  # unknown_behavior_indicator
        ]
    
    def generate_attack_batch(self, batch_size: int = 32,
                             attack_types: Optional[List[str]] = None) -> List[Dict]:
        """Generate batch of attack patterns"""
        if attack_types is None:
            attack_types = list(self.attack_types.keys())
        
        batch = []
        for _ in range(batch_size):
            attack_type = np.random.choice(attack_types)
            attack_pattern = self.attack_types[attack_type]()
            batch.append(attack_pattern)
        
        return batch
    
    def add_real_time_feed(self, feed_url: str) -> None:
        """Add real-time attack feed (IDS/IPS data)"""
        self.real_time_feeds.append({
            'url': feed_url,
            'active': True,
            'last_update': datetime.now().isoformat()
        })
    
    def fetch_real_time_attacks(self, limit: int = 100) -> List[Dict]:
        """Fetch real attacks from live feeds"""
        # In production, would fetch from actual feeds
        # For now, generate synthetic samples
        return self.generate_attack_batch(limit)


# ============================================================================
# CYBERSECURITY TRAINER
# ============================================================================

class CybersecurityTrainer(ReflectorIntegratedTrainer):
    """
    Specialized trainer for cybersecurity attack detection and mitigation.
    Includes offense-based learning for better defense.
    """
    
    def initialize(self) -> None:
        super().initialize()
        
        self.attack_generator = AttackPatternGenerator()
        self.attack_knowledge_base = {}
        self.defense_strategies = {}
        self.evasion_techniques = {}
        
        # Add cybersecurity-specific metrics
        self.training_history.update({
            'attack_detection_rate': [],
            'false_positive_rate': [],
            'evasion_success_rate': [],
            'attack_coverage': []
        })
        
        # Attack type weights for balanced training
        self.attack_weights = {attack_type: 1.0 
                              for attack_type in self.attack_generator.attack_types.keys()}
    
    def generate_training_data(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training data with attack patterns"""
        attacks = self.attack_generator.generate_attack_batch(batch_size)
        
        # Extract features and labels
        features = []
        labels = []
        
        for attack in attacks:
            attack_type = attack['type']
            features.append(attack['detection_features'])
            
            # Create label: 1 for attack, 0 for benign
            labels.append(1.0)
            
            # Track attack
            if attack_type not in self.attack_knowledge_base:
                self.attack_knowledge_base[attack_type] = {
                    'count': 0,
                    'patterns': [],
                    'severity': attack['severity']
                }
            
            self.attack_knowledge_base[attack_type]['count'] += 1
            self.attack_knowledge_base[attack_type]['patterns'].append(attack)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        return features_tensor, labels_tensor
    
    def generate_benign_data(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate benign (non-attack) training data"""
        # Create normal traffic patterns
        features = []
        labels = []
        
        for _ in range(batch_size):
            # Random normal features with lower anomaly scores
            feature = [np.random.uniform(0, 0.3) for _ in range(4)]
            features.append(feature)
            labels.append(0.0)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        return features_tensor, labels_tensor
    
    def train_step_cybersec(self, batch: torch.Tensor,
                           labels: torch.Tensor) -> Dict[str, float]:
        """
        Cybersecurity-specific training step with attack adversarialism
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(batch)
        
        # Primary loss
        primary_loss = self.loss_fn(output, labels)
        
        # Adversarial loss: model should correctly identify evasion attempts
        with torch.no_grad():
            # Generate adversarial examples
            adversarial_batch = self._generate_adversarial_examples(batch)
        
        adversarial_output = self.model(adversarial_batch)
        adversarial_loss = nn.BCEWithLogitsLoss()(
            adversarial_output, labels
        )
        
        # Reflector loss
        reflector_loss = torch.tensor(
            self.get_reflector_loss(output, labels)
        )
        
        # Combined loss: primary + adversarial + reflector
        total_loss = (
            0.5 * primary_loss +
            0.3 * adversarial_loss +
            self.reflector_weight * reflector_loss
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        metrics = {
            'primary_loss': primary_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'reflector_loss': reflector_loss.item(),
            'total_loss': total_loss.item(),
            'batch_size': batch.shape[0]
        }
        
        self.training_history['loss'].append(metrics['primary_loss'])
        
        return metrics
    
    def _generate_adversarial_examples(self, batch: torch.Tensor,
                                       epsilon: float = 0.1) -> torch.Tensor:
        """Generate adversarial examples to test evasion"""
        adversarial_batch = batch.clone().detach().requires_grad_(True)
        
        # FGSM-style attack
        output = self.model(adversarial_batch)
        loss = output.mean()
        
        if adversarial_batch.grad is not None:
            adversarial_batch.grad.zero_()
        
        loss.backward()
        
        if adversarial_batch.grad is not None:
            adversarial_batch = adversarial_batch + epsilon * adversarial_batch.grad.sign()
        
        return adversarial_batch.detach()
    
    def evaluate_attack_detection(self, test_attacks: List[Dict],
                                 test_benign: List[Dict]) -> Dict[str, float]:
        """Evaluate attack detection performance"""
        self.model.eval()
        
        # Prepare test data
        attack_features = torch.tensor(
            [a['detection_features'] for a in test_attacks],
            dtype=torch.float32
        )
        benign_features = torch.tensor(
            [b for b in test_benign],
            dtype=torch.float32
        )
        
        with torch.no_grad():
            attack_predictions = self.model(attack_features)
            benign_predictions = self.model(benign_features)
        
        # Calculate metrics
        attack_detection_rate = (attack_predictions > 0.5).float().mean().item()
        false_positive_rate = (benign_predictions > 0.5).float().mean().item()
        
        metrics = {
            'attack_detection_rate': attack_detection_rate,
            'false_positive_rate': false_positive_rate,
            'f1_score': 2 * (attack_detection_rate * (1 - false_positive_rate)) / 
                       (attack_detection_rate + (1 - false_positive_rate) + 1e-8)
        }
        
        self.training_history['attack_detection_rate'].append(attack_detection_rate)
        self.training_history['false_positive_rate'].append(false_positive_rate)
        
        self.model.train()
        
        return metrics
    
    def generate_defense_strategy(self, attack_type: str) -> Dict[str, Any]:
        """Generate defense strategy for attack type"""
        if attack_type not in self.attack_knowledge_base:
            return {}
        
        knowledge = self.attack_knowledge_base[attack_type]
        
        strategy = {
            'attack_type': attack_type,
            'frequency': knowledge['count'],
            'severity': knowledge['severity'],
            'detection_rules': self._generate_detection_rules(attack_type),
            'mitigation_steps': self._generate_mitigation(attack_type),
            'patches': self._suggest_patches(attack_type)
        }
        
        self.defense_strategies[attack_type] = strategy
        
        return strategy
    
    @staticmethod
    def _generate_detection_rules(attack_type: str) -> List[str]:
        """Generate detection rules for attack type"""
        rules = {
            'sql_injection': [
                'Monitor for SQL keywords in user inputs',
                'Check for unusual quotes and semicolons',
                'Analyze query structure anomalies'
            ],
            'xss': [
                'Filter <script> tags and event handlers',
                'Encode HTML entities',
                'Implement CSP headers'
            ],
            'ddos': [
                'Monitor request rate anomalies',
                'Track source IP diversity',
                'Detect packet pattern similarities'
            ],
            'buffer_overflow': [
                'Check buffer sizes before copy operations',
                'Monitor for large offset values',
                'Detect NOP sled patterns'
            ]
        }
        
        return rules.get(attack_type, [])
    
    @staticmethod
    def _generate_mitigation(attack_type: str) -> List[str]:
        """Generate mitigation strategies"""
        mitigations = {
            'sql_injection': [
                'Use prepared statements',
                'Implement input validation',
                'Apply principle of least privilege'
            ],
            'xss': [
                'Implement output encoding',
                'Use security-focused templating',
                'Apply CSP restrictions'
            ],
            'ddos': [
                'Deploy rate limiting',
                'Use CDN and traffic filtering',
                'Implement DDoS mitigation service'
            ]
        }
        
        return mitigations.get(attack_type, [])
    
    @staticmethod
    def _suggest_patches(attack_type: str) -> List[str]:
        """Suggest security patches"""
        # In production, would query CVE databases
        return [
            'Update to latest security patch',
            'Review vendor advisories',
            'Test in staging environment first'
        ]


def train_step_cybersec(self, batch):
    with torch.set_grad_enabled(True):
        # Train on batch
        return loss

def train_step_cybersec(self, batch):
    if len(batch) != self.batch_size:
        raise ValueError("Batch size mismatch")
    # training step code