"""
Emergence detection module for the Prompt Theory framework.

This module implements the emergence detection mechanisms defined in the Prompt Theory
mathematical framework, providing tools for identifying, measuring, and characterizing
emergent properties in both AI and human cognitive systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import warnings
from enum import Enum
import networkx as nx

from prompt_theory.models.recursion import RecursiveProcessor
from prompt_theory.utils.logging import get_logger
from prompt_theory.utils.validation import validate_parameters


logger = get_logger(__name__)


class EmergenceType(Enum):
    """Types of emergence that can occur in information processing systems."""
    
    WEAK = "weak"  # Properties derivable from components but not obvious
    STRONG = "strong"  # Properties not derivable from components
    NOVEL_BEHAVIOR = "novel_behavior"  # New behavioral patterns
    NOVEL_CAPABILITY = "novel_capability"  # New system capabilities
    PATTERN_FORMATION = "pattern_formation"  # Organized structures
    AUTOPOIESIS = "autopoiesis"  # Self-maintaining structures
    INTERACTIVE = "interactive"  # Emergent properties from interactions
    PHASE_TRANSITION = "phase_transition"  # Sudden qualitative changes


class EmergenceDetector:
    """Detects and analyzes emergence in recursive systems.
    
    This class implements the emergence detection mechanisms defined in
    the Prompt Theory mathematical framework, focusing on the conditions
    under which new properties, behaviors, or capabilities emerge from
    recursive processing.
    
    Attributes:
        recursive_processor: Model for recursive processing
        emergence_threshold: Threshold for emergence detection
        integration_threshold: Threshold for integrated information
        _state: Internal state tracking emergence history
    """
    
    def __init__(
        self,
        recursive_processor: Optional[RecursiveProcessor] = None,
        emergence_threshold: float = 0.6,
        integration_threshold: float = 0.7,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize emergence detector.
        
        Args:
            recursive_processor: Recursive processing model
            emergence_threshold: Threshold for emergence detection (0-1). Default 0.6.
            integration_threshold: Threshold for integrated information (0-1). Default 0.7.
            config: Additional configuration parameters
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        validate_parameters(
            ("emergence_threshold", emergence_threshold, 0.0, 1.0),
            ("integration_threshold", integration_threshold, 0.0, 1.0),
        )
        
        # Set configuration
        self.config = config or {}
        
        # Initialize recursive processor if not provided
        self.recursive_processor = recursive_processor or RecursiveProcessor(
            max_depth=self.config.get("max_recursion_depth", 4),
            collapse_threshold=self.config.get("collapse_threshold", 0.8),
            emergence_threshold=emergence_threshold,
        )
        
        # Store thresholds
        self.emergence_threshold = emergence_threshold
        self.integration_threshold = integration_threshold
        
        # Initialize state
        self._state = {
            "emergence_history": [],
            "current_emergence": None,
            "emergence_count": 0,
            "emergence_types": {},
        }
        
        logger.debug(f"Initialized EmergenceDetector with thresholds: emergence={emergence_threshold}, integration={integration_threshold}")
    
    def detect_emergence(
        self,
        state_history: List[Dict[str, Any]],
        analysis_depth: str = "standard",
        detect_type: bool = True,
    ) -> Dict[str, Any]:
        """Detect emergence based on state history.
        
        This method analyzes system state history to identify emergent
        properties, behaviors, or capabilities according to the conditions
        defined in Section 4.4 of the Prompt Theory mathematical framework.
        
        Args:
            state_history: History of system states
            analysis_depth: Depth of analysis ('minimal', 'standard', 'comprehensive')
            detect_type: Whether to detect emergence type
            
        Returns:
            Dictionary with emergence detection results
            
        Raises:
            ValueError: If state_history is invalid or empty
        """
        if not state_history:
            raise ValueError("State history cannot be empty")
        
        # Extract metrics from state history
        integration_values = []
        discontinuity_values = []
        
        for state in state_history:
            if "integration" in state:
                integration_values.append(state["integration"])
            if "discontinuity" in state:
                discontinuity_values.append(state["discontinuity"])
        
        # If required metrics are missing, use recursive processor to compute them
        if not integration_values or not discontinuity_values:
            processor_result = self.recursive_processor.detect_emergence(state_history)
            emergence_detected = processor_result.get("emergence_detected", False)
            confidence = processor_result.get("confidence", 0.0)
            indicators = processor_result.get("indicators", [])
        else:
            # Check for emergence conditions
            # 1. Integration exceeds threshold
            # 2. Discontinuity remains below collapse threshold
            max_integration = max(integration_values) if integration_values else 0.0
            corresponding_discontinuity = 0.0
            
            if integration_values:
                idx = integration_values.index(max_integration)
                if idx < len(discontinuity_values):
                    corresponding_discontinuity = discontinuity_values[idx]
            
            emergence_detected = (
                max_integration > self.emergence_threshold and
                corresponding_discontinuity < self.recursive_processor.parameters.collapse_threshold
            )
            
            # Calculate confidence based on how much integration exceeds threshold
            # and how far discontinuity is from collapse threshold
            if emergence_detected:
                integration_margin = (max_integration - self.emergence_threshold) / (1.0 - self.emergence_threshold)
                discontinuity_margin = (self.recursive_processor.parameters.collapse_threshold - corresponding_discontinuity) / self.recursive_processor.parameters.collapse_threshold
                confidence = min(1.0, (integration_margin + discontinuity_margin) / 2)
            else:
                confidence = 0.0
                
            # Create indicators
            indicators = []
            if max_integration > self.emergence_threshold:
                indicators.append({
                    "type": "high_integration",
                    "value": max_integration,
                    "threshold": self.emergence_threshold,
                    "confidence": (max_integration - self.emergence_threshold) / (1.0 - self.emergence_threshold),
                })
            
            if corresponding_discontinuity < self.recursive_processor.parameters.collapse_threshold:
                indicators.append({
                    "type": "stable_discontinuity",
                    "value": corresponding_discontinuity,
                    "threshold": self.recursive_processor.parameters.collapse_threshold,
                    "confidence": (self.recursive_processor.parameters.collapse_threshold - corresponding_discontinuity) / self.recursive_processor.parameters.collapse_threshold,
                })
        
        # Identify emergence type if requested
        emergence_type = None
        type_confidence = 0.0
        type_evidence = []
        
        if emergence_detected and detect_type:
            emergence_type, type_confidence, type_evidence = self._determine_emergence_type(
                state_history, analysis_depth
            )
        
        # Prepare emergence pattern analysis based on analysis depth
        pattern_analysis = None
        if emergence_detected and analysis_depth != "minimal":
            pattern_analysis = self._analyze_emergence_pattern(
                state_history, analysis_depth
            )
        
        # Update internal state
        if emergence_detected:
            self._state["emergence_count"] += 1
            self._state["emergence_history"].append({
                "time": len(self._state["emergence_history"]),
                "confidence": confidence,
                "type": emergence_type,
                "type_confidence": type_confidence,
            })
            
            if emergence_type:
                if emergence_type.value not in self._state["emergence_types"]:
                    self._state["emergence_types"][emergence_type.value] = 0
                self._state["emergence_types"][emergence_type.value] += 1
            
            self._state["current_emergence"] = {
                "confidence": confidence,
                "type": emergence_type,
                "type_confidence": type_confidence,
            }
        
        # Prepare result
        result = {
            "emergence_detected": emergence_detected,
            "confidence": confidence,
            "indicators": indicators,
        }
        
        if emergence_detected and detect_type:
            result["emergence_type"] = emergence_type.value if emergence_type else None
            result["type_confidence"] = type_confidence
            result["type_evidence"] = type_evidence
        
        if pattern_analysis:
            result["pattern_analysis"] = pattern_analysis
        
        return result
    
    def _determine_emergence_type(
        self,
        state_history: List[Dict[str, Any]],
        analysis_depth: str,
    ) -> Tuple[Optional[EmergenceType], float, List[Dict[str, Any]]]:
        """Determine the type of emergence based on state history patterns."""
        # Extract data from state history
        integration_values = [state.get("integration", 0.0) for state in state_history if "integration" in state]
        discontinuity_values = [state.get("discontinuity", 0.0) for state in state_history if "discontinuity" in state]
        
        # Evidence collection for each emergence type
        evidence = {etype: [] for etype in EmergenceType}
        
        # Check for weak emergence
        if self._check_weak_emergence(state_history):
            evidence[EmergenceType.WEAK].append({
                "description": "System exhibits properties derivable from components",
                "confidence": 0.8,
            })
        
        # Check for strong emergence
        if self._check_strong_emergence(state_history):
            evidence[EmergenceType.STRONG].append({
                "description": "System exhibits properties not derivable from components",
                "confidence": 0.7,
            })
        
        # Check for novel behavior emergence
        if self._check_novel_behavior(state_history):
            evidence[EmergenceType.NOVEL_BEHAVIOR].append({
                "description": "System exhibits new behavioral patterns",
                "confidence": 0.75,
            })
        
        # Check for novel capability emergence
        if self._check_novel_capability(state_history):
            evidence[EmergenceType.NOVEL_CAPABILITY].append({
                "description": "System exhibits new capabilities",
                "confidence": 0.8,
            })
        
        # Check for pattern formation
        if self._check_pattern_formation(state_history):
            evidence[EmergenceType.PATTERN_FORMATION].append({
                "description": "System exhibits organized structural patterns",
                "confidence": 0.7,
            })
        
        # Check for autopoiesis (self-maintenance)
        if analysis_depth == "comprehensive" and self._check_autopoiesis(state_history):
            evidence[EmergenceType.AUTOPOIESIS].append({
                "description": "System exhibits self-maintaining structures",
                "confidence": 0.6,
            })
        
        # Check for interactive emergence
        if analysis_depth == "comprehensive" and self._check_interactive_emergence(state_history):
            evidence[EmergenceType.INTERACTIVE].append({
                "description": "System exhibits emergence from component interactions",
                "confidence": 0.7,
            })
        
        # Check for phase transition
        if self._check_phase_transition(integration_values, discontinuity_values):
            evidence[EmergenceType.PHASE_TRANSITION].append({
                "description": "System exhibits sudden qualitative changes",
                "confidence": 0.8,
            })
        
        # Determine most likely emergence type
        max_confidence = 0.0
        best_type = None
        best_evidence = []
        
        for etype, type_evidence in evidence.items():
            if type_evidence:
                confidence = sum(e["confidence"] for e in type_evidence) / len(type_evidence)
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_type = etype
                    best_evidence = type_evidence
        
        return best_type, max_confidence, best_evidence
    
    def _check_weak_emergence(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for weak emergence - properties derivable from components but not obvious."""
        # Weak emergence is the default type when emergence is detected
        # but no stronger forms are identified
        return True
    
    def _check_strong_emergence(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for strong emergence - properties not derivable from components."""
        # Look for significant jumps in integration without corresponding
        # jumps in component metrics
        
        integration_values = [state.get("integration", 0.0) for state in state_history if "integration" in state]
        
        if len(integration_values) < 2:
            return False
        
        # Check for significant jumps in integration
        for i in range(1, len(integration_values)):
            jump = integration_values[i] - integration_values[i-1]
            if jump > 0.3:  # Significant jump threshold
                # Check if component metrics also jumped
                component_jump = False
                
                # Look for component metrics in state
                for key in state_history[i].keys():
                    if key.startswith("component_") and key in state_history[i-1]:
                        prev_value = state_history[i-1][key]
                        curr_value = state_history[i][key]
                        
                        if isinstance(prev_value, (int, float)) and isinstance(curr_value, (int, float)):
                            comp_jump = abs(curr_value - prev_value) / max(0.001, abs(prev_value))
                            if comp_jump > 0.3:  # Significant component jump
                                component_jump = True
                                break
                
                # If integration jumped but components didn't, that's strong emergence
                if not component_jump:
                    return True
        
        return False
    
    def _check_novel_behavior(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for novel behavior emergence - new behavioral patterns."""
        # Look for new behavioral patterns that weren't present in earlier states
        
        # Simplified check: look for new keys in state that indicate new behaviors
        if len(state_history) < 2:
            return False
        
        # Collect keys from early states (first third)
        early_idx = max(1, len(state_history) // 3)
        early_keys = set()
        for i in range(early_idx):
            for key in state_history[i].keys():
                if key.startswith("behavior_") or key.startswith("action_"):
                    early_keys.add(key)
        
        # Check for new behavior keys in later states
        for i in range(early_idx, len(state_history)):
            for key in state_history[i].keys():
                if (key.startswith("behavior_") or key.startswith("action_")) and key not in early_keys:
                    # Found new behavior key
                    return True
        
        return False
    
    def _check_novel_capability(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for novel capability emergence - new system capabilities."""
        # Look for new capabilities that weren't present in earlier states
        
        # Simplified check: look for new keys in state that indicate new capabilities
        if len(state_history) < 2:
            return False
        
        # Collect keys from early states (first third)
        early_idx = max(1, len(state_history) // 3)
        early_keys = set()
        for i in range(early_idx):
            for key in state_history[i].keys():
                if key.startswith("capability_") or key.startswith("can_"):
                    early_keys.add(key)
        
        # Check for new capability keys in later states
        for i in range(early_idx, len(state_history)):
            for key in state_history[i].keys():
                if (key.startswith("capability_") or key.startswith("can_")) and key not in early_keys:
                    # Found new capability key
                    return True
        
        return False
    
    def _check_pattern_formation(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for pattern formation emergence - organized structures."""
        # Look for organized patterns in state variables
        
        # Simplified check: look for repeating patterns in state values
        if len(state_history) < 4:  # Need at least a few states to detect patterns
            return False
        
        # Extract numerical sequences for pattern detection
        sequences = {}
        for key in state_history[0].keys():
            if key not in ["integration", "discontinuity"]:  # Skip core metrics
                sequence = []
                for state in state_history:
                    if key in state and isinstance(state[key], (int, float)):
                        sequence.append(state[key])
                
                if len(sequence) >= 4:  # Need at least a few values
                    sequences[key] = sequence
        
        # Check each sequence for patterns
        for key, sequence in sequences.items():
            # Check for oscillation pattern
            if self._detect_oscillation(sequence):
                return True
            
            # Check for trend pattern
            if self._detect_trend(sequence):
                return True
        
        return False
    
    def _check_autopoiesis(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for autopoiesis emergence - self-maintaining structures."""
        # Look for structures that maintain themselves over time
        
        # Simplified check: look for stable structures despite external perturbations
        if len(state_history) < 5:  # Need several states to detect stability
            return False
        
        # Look for perturbation indicators
        perturbations = []
        for i in range(1, len(state_history)):
            if "perturbation" in state_history[i] or "disturbance" in state_history[i]:
                perturbations.append(i)
        
        if not perturbations:
            return False  # No perturbations detected
        
        # Check if certain structures remain stable despite perturbations
        stable_structures = []
        
        for key in state_history[0].keys():
            if key.startswith("structure_") or key.startswith("component_"):
                stable = True
                
                # Check stability across perturbations
                for p_idx in perturbations:
                    if p_idx < len(state_history) - 1:  # Ensure there's a state after perturbation
                        before_value = state_history[p_idx-1].get(key)
                        after_value = state_history[p_idx+1].get(key)
                        
                        if before_value != after_value:
                            stable = False
                            break
                
                if stable:
                    stable_structures.append(key)
        
        return len(stable_structures) > 0
    
    def _check_interactive_emergence(self, state_history: List[Dict[str, Any]]) -> bool:
        """Check for interactive emergence - emergence from component interactions."""
        # Look for emergence that results from interactions between components
        
        # Simplified check: look for interaction metrics that correlate with integration
        if len(state_history) < 3:
            return False
        
        # Extract integration values and interaction metrics
        integration_values = []
        interaction_metrics = {}
        
        for state in state_history:
            if "integration" in state:
                integration = state["integration"]
                integration_values.append(integration)
                
                # Collect interaction metrics
                for key, value in state.items():
                    if (key.startswith("interaction_") or key.startswith("connection_")) and isinstance(value, (int, float)):
                        if key not in interaction_metrics:
