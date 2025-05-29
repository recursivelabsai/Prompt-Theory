"""
Drift and stability model for the Prompt Theory framework.

This module implements the mathematical models for drift and stability in both AI and
human cognitive systems, providing a unified framework for understanding and managing
change over time in recursive information processing systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import warnings
from enum import Enum

from prompt_theory.utils.math import cosine_similarity, angular_distance
from prompt_theory.utils.logging import get_logger
from prompt_theory.utils.validation import validate_parameters


logger = get_logger(__name__)


class DriftType(Enum):
    """Types of drift that can occur in information processing systems."""
    
    INTENTIONAL = "intentional"  # Purposeful, goal-aligned changes
    UNINTENTIONAL = "unintentional"  # Unplanned, potentially problematic changes
    ADAPTIVE = "adaptive"  # Beneficial changes in response to context
    MALADAPTIVE = "maladaptive"  # Harmful changes in response to context
    OSCILLATORY = "oscillatory"  # Repeated back-and-forth changes
    DIVERGENT = "divergent"  # Continuously increasing deviation
    CONVERGENT = "convergent"  # Changes that approach a stable state


@dataclass
class StabilityParameters:
    """Parameters controlling stability and drift behavior."""
    
    anchoring_weight: float = 0.7  # Weight for anchoring to reference state (0-1)
    drift_detection_threshold: float = 0.3  # Threshold for detecting drift (0-1)
    stabilization_rate: float = 0.5  # Rate of active stabilization (0-1)
    adaptive_tolerance: float = 0.2  # Tolerance for adaptive changes (0-1)
    goal_alignment_weight: float = 0.6  # Weight for goal-aligned changes (0-1)


class DriftModel:
    """Models drift and stability in recursive systems.
    
    This class implements the drift and stability mechanisms defined in the
    Prompt Theory mathematical framework, providing tools for measuring,
    predicting, and managing drift in information processing systems.
    
    Attributes:
        parameters: StabilityParameters controlling drift and stability behavior
        _state: Internal state tracking drift history and stability metrics
    """
    
    def __init__(
        self,
        stability_params: Optional[Dict[str, float]] = None,
        drift_detection_threshold: float = 0.3,
    ):
        """Initialize drift model.
        
        Args:
            stability_params: Parameters for stability calculation
            drift_detection_threshold: Threshold for drift detection (0-1). Default 0.3.
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Apply default or provided stability parameters
        params = stability_params or {}
        
        self.parameters = StabilityParameters(
            anchoring_weight=params.get("anchoring_weight", 0.7),
            drift_detection_threshold=drift_detection_threshold,
            stabilization_rate=params.get("stabilization_rate", 0.5),
            adaptive_tolerance=params.get("adaptive_tolerance", 0.2),
            goal_alignment_weight=params.get("goal_alignment_weight", 0.6),
        )
        
        # Validate parameter ranges
        validate_parameters(
            ("anchoring_weight", self.parameters.anchoring_weight, 0.0, 1.0),
            ("drift_detection_threshold", self.parameters.drift_detection_threshold, 0.0, 1.0),
            ("stabilization_rate", self.parameters.stabilization_rate, 0.0, 1.0),
            ("adaptive_tolerance", self.parameters.adaptive_tolerance, 0.0, 1.0),
            ("goal_alignment_weight", self.parameters.goal_alignment_weight, 0.0, 1.0),
        )
        
        self._state = {
            "drift_history": [],
            "reference_state": None,
            "current_drift": 0.0,
            "drift_vector": None,
            "drift_type": None,
            "stability_metric": 1.0,
        }
        
        logger.debug(f"Initialized DriftModel with parameters: {self.parameters}")
    
    def measure_drift(
        self,
        state_history: List[Dict[str, Any]],
        reference_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Measure drift from reference state.
        
        This method implements the drift function Δ(St, S0) defined in Section 4.5
        of the Prompt Theory mathematical framework.
        
        Args:
            state_history: History of system states
            reference_state: Reference state for drift calculation. If None, the first
                state in history is used.
            metrics: List of specific metrics to measure. If None, all available metrics
                are measured.
        
        Returns:
            Dictionary containing drift measurements and analysis
        
        Raises:
            ValueError: If state_history is empty
        """
        if not state_history:
            raise ValueError("State history cannot be empty")
        
        # Set reference state
        reference = reference_state or state_history[0]
        
        # Initialize drift metrics
        drift_metrics = {}
        
        # Calculate cumulative drift
        cumulative_drift = self._calculate_cumulative_drift(state_history, reference)
        drift_metrics["cumulative"] = cumulative_drift
        
        # Calculate incremental drifts
        incremental_drifts = self._calculate_incremental_drifts(state_history)
        drift_metrics["incremental"] = incremental_drifts
        
        # Calculate drift velocity (rate of change)
        drift_velocity = self._calculate_drift_velocity(incremental_drifts)
        drift_metrics["velocity"] = drift_velocity
        
        # Calculate drift acceleration (change in velocity)
        drift_acceleration = self._calculate_drift_acceleration(incremental_drifts)
        drift_metrics["acceleration"] = drift_acceleration
        
        # Determine drift type
        drift_type = self._determine_drift_type(
            incremental_drifts, cumulative_drift, reference
        )
        
        # Calculate drift vector if states contain vector representations
        drift_vector = self._calculate_drift_vector(state_history, reference)
        
        # Determine if drift exceeds threshold
        significant_drift = cumulative_drift > self.parameters.drift_detection_threshold
        
        # Update internal state
        self._state["drift_history"].append({
            "time": len(self._state["drift_history"]),
            "cumulative": cumulative_drift,
            "incremental": incremental_drifts[-1] if incremental_drifts else 0.0,
            "velocity": drift_velocity,
            "acceleration": drift_acceleration,
            "type": drift_type,
        })
        
        self._state["reference_state"] = reference
        self._state["current_drift"] = cumulative_drift
        self._state["drift_vector"] = drift_vector
        self._state["drift_type"] = drift_type
        
        # Prepare result
        result = {
            "metrics": drift_metrics,
            "significant_drift": significant_drift,
            "drift_type": drift_type,
            "drift_vector": drift_vector,
            "analysis": self._analyze_drift(
                cumulative_drift, incremental_drifts, drift_velocity, drift_acceleration, drift_type
            ),
        }
        
        return result
    
    def _calculate_cumulative_drift(
        self, state_history: List[Dict[str, Any]], reference_state: Dict[str, Any]
    ) -> float:
        """Calculate cumulative drift from reference state.
        
        This implements the core drift function Δ(St, S0) defined in Section 4.5
        of the Prompt Theory mathematical framework.
        """
        if len(state_history) <= 1:
            return 0.0
        
        # Get current state (most recent in history)
        current_state = state_history[-1]
        
        # Calculate state distance
        return self._calculate_state_distance(current_state, reference_state)
    
    def _calculate_incremental_drifts(
        self, state_history: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate incremental drifts between consecutive states."""
        if len(state_history) <= 1:
            return [0.0]
        
        incremental_drifts = []
        for i in range(1, len(state_history)):
            drift = self._calculate_state_distance(state_history[i], state_history[i-1])
            incremental_drifts.append(drift)
        
        return incremental_drifts
    
    def _calculate_drift_velocity(self, incremental_drifts: List[float]) -> float:
        """Calculate drift velocity (rate of change)."""
        if not incremental_drifts:
            return 0.0
        
        # Use recent history for velocity calculation
        window_size = min(5, len(incremental_drifts))
        recent_drifts = incremental_drifts[-window_size:]
        
        return sum(recent_drifts) / window_size
    
    def _calculate_drift_acceleration(self, incremental_drifts: List[float]) -> float:
        """Calculate drift acceleration (change in velocity)."""
        if len(incremental_drifts) < 2:
            return 0.0
        
        # Calculate differences between consecutive incremental drifts
        drift_changes = [
            incremental_drifts[i] - incremental_drifts[i-1]
            for i in range(1, len(incremental_drifts))
        ]
        
        # Use recent history for acceleration calculation
        window_size = min(5, len(drift_changes))
        recent_changes = drift_changes[-window_size:]
        
        return sum(recent_changes) / window_size
    
    def _determine_drift_type(
        self,
        incremental_drifts: List[float],
        cumulative_drift: float,
        reference_state: Dict[str, Any],
    ) -> DriftType:
        """Determine the type of drift based on drift patterns."""
        if not incremental_drifts:
            return DriftType.INTENTIONAL  # Default if no drift detected
        
        # Check for oscillatory drift
        if len(incremental_drifts) >= 4:
            # Look for alternating positive and negative incremental drifts
            sign_changes = sum(
                1 for i in range(1, len(incremental_drifts))
                if (incremental_drifts[i] > 0 and incremental_drifts[i-1] < 0) or
                   (incremental_drifts[i] < 0 and incremental_drifts[i-1] > 0)
            )
            if sign_changes >= len(incremental_drifts) // 2:
                return DriftType.OSCILLATORY
        
        # Check for divergent drift
        if len(incremental_drifts) >= 3:
            # Look for consistently increasing incremental drifts
            consistently_increasing = all(
                incremental_drifts[i] >= incremental_drifts[i-1]
                for i in range(1, len(incremental_drifts))
            )
            if consistently_increasing and incremental_drifts[-1] > incremental_drifts[0] * 1.5:
                return DriftType.DIVERGENT
        
        # Check for convergent drift
        if len(incremental_drifts) >= 3:
            # Look for consistently decreasing incremental drifts
            consistently_decreasing = all(
                incremental_drifts[i] <= incremental_drifts[i-1]
                for i in range(1, len(incremental_drifts))
            )
            if consistently_decreasing and incremental_drifts[-1] < incremental_drifts[0] * 0.5:
                return DriftType.CONVERGENT
        
        # Default to intentional vs. unintentional based on threshold
        if cumulative_drift <= self.parameters.adaptive_tolerance:
            return DriftType.INTENTIONAL
        else:
            return DriftType.UNINTENTIONAL
    
    def _calculate_drift_vector(
        self, state_history: List[Dict[str, Any]], reference_state: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Calculate drift vector if states contain vector representations."""
        if not state_history:
            return None
        
        current_state = state_history[-1]
        
        # Check if states contain vector representations
        if "vector" in current_state and "vector" in reference_state:
            try:
                current_vector = np.array(current_state["vector"])
                reference_vector = np.array(reference_state["vector"])
                
                # Calculate vector difference
                return current_vector - reference_vector
            except (ValueError, TypeError):
                logger.warning("Failed to calculate drift vector from state vectors")
                return None
        
        return None
    
    def _analyze_drift(
        self,
        cumulative_drift: float,
        incremental_drifts: List[float],
        drift_velocity: float,
        drift_acceleration: float,
        drift_type: DriftType,
    ) -> Dict[str, Any]:
        """Analyze drift patterns and provide insights."""
        analysis = {
            "severity": self._analyze_drift_severity(cumulative_drift),
            "trajectory": self._analyze_drift_trajectory(
                drift_velocity, drift_acceleration, drift_type
            ),
            "stability_projection": self._project_stability(
                cumulative_drift, drift_velocity, drift_acceleration, drift_type
            ),
            "recommendations": self._generate_recommendations(
                cumulative_drift, drift_velocity, drift_acceleration, drift_type
            ),
        }
        
        return analysis
    
    def _analyze_drift_severity(self, cumulative_drift: float) -> Dict[str, Any]:
        """Analyze the severity of detected drift."""
        # Define severity thresholds
        low_threshold = self.parameters.drift_detection_threshold * 0.5
        medium_threshold = self.parameters.drift_detection_threshold
        high_threshold = self.parameters.drift_detection_threshold * 2.0
        
        # Determine severity level
        if cumulative_drift < low_threshold:
            severity = "low"
            description = "Minimal drift detected, within normal operating parameters."
            risk_level = "low"
        elif cumulative_drift < medium_threshold:
            severity = "moderate"
            description = "Noticeable drift detected, approaching intervention threshold."
            risk_level = "moderate"
        elif cumulative_drift < high_threshold:
            severity = "high"
            description = "Significant drift detected, exceeding intervention threshold."
            risk_level = "high"
        else:
            severity = "critical"
            description = "Critical drift detected, system integrity at risk."
            risk_level = "critical"
        
        return {
            "level": severity,
            "description": description,
            "risk_level": risk_level,
            "threshold_ratio": cumulative_drift / self.parameters.drift_detection_threshold,
        }
    
    def _analyze_drift_trajectory(
        self, drift_velocity: float, drift_acceleration: float, drift_type: DriftType
    ) -> Dict[str, Any]:
        """Analyze the trajectory of drift based on velocity and acceleration."""
        # Determine trajectory characteristics
        if drift_velocity < 0.01:
            direction = "stable"
            description = "System is maintaining stability with minimal drift."
        elif drift_velocity < 0.05:
            direction = "slow drift"
            description = "System is experiencing slow, controlled drift."
        elif drift_velocity < 0.1:
            direction = "moderate drift"
            description = "System is experiencing moderate drift."
        else:
            direction = "rapid drift"
            description = "System is experiencing rapid drift."
        
        # Analyze acceleration
        if abs(drift_acceleration) < 0.01:
            trend = "steady"
            trend_description = "Drift rate is steady."
        elif drift_acceleration > 0:
            trend = "accelerating"
            trend_description = "Drift rate is increasing."
        else:
            trend = "decelerating"
            trend_description = "Drift rate is decreasing."
        
        return {
            "direction": direction,
            "description": description,
            "trend": trend,
            "trend_description": trend_description,
            "drift_type": drift_type.value,
        }
    
    def _project_stability(
        self,
        cumulative_drift: float,
        drift_velocity: float,
        drift_acceleration: float,
        drift_type: DriftType,
    ) -> Dict[str, Any]:
        """Project future stability based on current drift patterns."""
        # Calculate time to intervention threshold
        if drift_velocity <= 0:
            time_to_threshold = float('inf')  # Stable or improving
        else:
            distance_to_threshold = max(0, self.parameters.drift_detection_threshold - cumulative_drift)
            time_to_threshold = distance_to_threshold / drift_velocity if distance_to_threshold > 0 else 0
        
        # Determine stability projection
        if drift_type == DriftType.CONVERGENT or drift_velocity < 0:
            projection = "improving"
            description = "System stability is projected to improve."
        elif drift_type == DriftType.OSCILLATORY:
            projection = "fluctuating"
            description = "System stability is projected to fluctuate."
        elif drift_velocity < 0.01:
            projection = "stable"
            description = "System stability is projected to remain consistent."
        elif drift_acceleration < 0:
            projection = "stabilizing"
            description = "System drift is slowing and projected to stabilize."
        else:
            projection = "deteriorating"
            description = "System stability is projected to deteriorate."
        
        return {
            "projection": projection,
            "description": description,
            "time_to_threshold": time_to_threshold,
            "estimated_max_drift": self._estimate_max_drift(
                cumulative_drift, drift_velocity, drift_acceleration, drift_type
            ),
        }
    
    def _estimate_max_drift(
        self,
        cumulative_drift: float,
        drift_velocity: float,
        drift_acceleration: float,
        drift_type: DriftType,
    ) -> float:
        """Estimate maximum expected drift based on current patterns."""
        if drift_type == DriftType.CONVERGENT or drift_velocity <= 0:
            # Converging or improving
            return cumulative_drift
        elif drift_type == DriftType.OSCILLATORY:
            # Oscillating around a central value
            return cumulative_drift * 1.5
        elif drift_acceleration <= 0:
            # Linear drift with no acceleration
            return cumulative_drift + drift_velocity * 10
        else:
            # Accelerating drift
            return cumulative_drift + drift_velocity * 10 + 0.5 * drift_acceleration * 10 * 10
    
    def _generate_recommendations(
        self,
        cumulative_drift: float,
        drift_velocity: float,
        drift_acceleration: float,
        drift_type: DriftType,
    ) -> List[Dict[str, str]]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        # Recommendations based on drift magnitude
        if cumulative_drift > self.parameters.drift_detection_threshold:
            recommendations.append({
                "type": "intervention",
                "description": "Implement drift correction to return system toward reference state.",
                "priority": "high" if cumulative_drift > self.parameters.drift_detection_threshold * 2 else "medium",
            })
        elif cumulative_drift > self.parameters.drift_detection_threshold * 0.5:
            recommendations.append({
                "type": "monitoring",
                "description": "Increase monitoring frequency and set up alerts for further drift.",
                "priority": "medium",
            })
        
        # Recommendations based on drift velocity
        if drift_velocity > 0.1:
            recommendations.append({
                "type": "velocity_control",
                "description": "Implement measures to slow the rate of drift.",
                "priority": "high",
            })
        
        # Recommendations based on drift type
        if drift_type == DriftType.OSCILLATORY:
            recommendations.append({
                "type": "stabilization",
                "description": "Implement damping mechanisms to reduce oscillatory behavior.",
                "priority": "medium",
            })
        elif drift_type == DriftType.DIVERGENT:
            recommendations.append({
                "type": "convergence",
                "description": "Implement strong anchoring to reference state to prevent further divergence.",
                "priority": "high",
            })
        
        # If no specific recommendations, add general guidance
        if not recommendations:
            recommendations.append({
                "type": "maintenance",
                "description": "Maintain current stability measures and regular monitoring.",
                "priority": "low",
            })
        
        return recommendations
    
    def _calculate_state_distance(
        self, state1: Dict[str, Any], state2: Dict[str, Any]
    ) -> float:
        """Calculate distance between two states.
        
        This function supports different distance calculations based on state structure:
        - Vector-based distance if states contain vector representations
        - Feature-based distance for structured state dictionaries
        - Simple change ratio for unstructured states
        """
        # Check for vector representations
        if "vector" in state1 and "vector" in state2:
            try:
                vec1 = np.array(state1["vector"])
                vec2 = np.array(state2["vector"])
                return self._calculate_vector_distance(vec1, vec2)
            except (ValueError, TypeError):
                logger.debug("Failed to calculate vector distance, falling back to feature distance")
        
        # Calculate feature-based distance
        return self._calculate_feature_distance(state1, state2)
    
    def _calculate_vector_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        try:
            # Use cosine distance for high-dimensional vectors
            if len(vec1) > 10:
                return 1.0 - cosine_similarity(vec1, vec2)
            else:
                # Use normalized Euclidean distance for lower-dimensional vectors
                return np.linalg.norm(vec1 - vec2) / np.sqrt(len(vec1))
        except Exception as e:
            logger.warning(f"Error calculating vector distance: {e}")
            return 0.5  # Default to moderate distance on error
    
    def _calculate_feature_distance(
        self, state1: Dict[str, Any], state2: Dict[str, Any]
    ) -> float:
        """Calculate feature-based distance between two states."""
        # Extract features (non-internal keys)
        features1 = {k: v for k, v in state1.items() if not k.startswith("_") and k != "vector"}
        features2 = {k: v for k, v in state2.items() if not k.startswith("_") and k != "vector"}
        
        # If no comparable features, return moderate distance
        if not features1 and not features2:
            return 0.5
        
        # Collect all feature keys
        all_keys = set(features1.keys()).union(set(features2.keys()))
        
        # Count changes and calculate distance
        feature_changes = 0
        feature_count = len(all_keys)
        
        for key in all_keys:
            if key not in features1 or key not in features2:
                # Feature added or removed
                feature_changes += 1
            elif features1[key] != features2[key]:
                # Feature changed
                if isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                    # Numerical feature - calculate relative change
                    val1 = features1[key]
                    val2 = features2[key]
                    if val1 == 0 and val2 == 0:
                        change = 0.0
                    elif val1 == 0:
                        change = 1.0
                    else:
                        change = min(1.0, abs((val2 - val1) / abs(val1)))
                    feature_changes += change
                else:
                    # Non-numerical feature
                    feature_changes += 1
        
        # Normalize changes to 0-1 scale
        return feature_changes / max(1, feature_count)
    
    def decompose_drift(
        self,
        state_history: List[Dict[str, Any]],
        goal_vector: Optional[np.ndarray] = None,
        reference_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Decompose drift into intentional and unintentional components.
        
        This method implements the drift decomposition function defined in 
        Section 4.5 of the Prompt Theory mathematical framework:
        
        Δ(St, S0) = ΔI(St, S0) + ΔU(St, S0)
        
        Args:
            state_history: History of system states
            goal_vector: Vector representing goal direction
            reference_state: Reference state for drift calculation
            
        Returns:
            Dictionary with decomposed drift measurements
            
        Raises:
            ValueError: If state_history is empty or goal_vector is invalid
        """
        if not state_history:
            raise ValueError("State history cannot be empty")
        
        # Set reference state
        reference = reference_state or state_history[0]
        
        # Calculate total drift
        total_drift = self._calculate_cumulative_drift(state_history, reference)
        
        # If no goal vector provided, assume all drift is unintentional
        if goal_vector is None:
            return {
                "total": total_drift,
                "intentional": 0.0,
                "unintentional": total_drift,
            }
        
        # Calculate drift vector
        drift_vector = self._calculate_drift_vector(state_history, reference)
        
        # If no drift vector available, use heuristic decomposition
        if drift_vector is None:
            # Default heuristic: 70% of drift within threshold is intentional
            if total_drift <= self.parameters.drift_detection_threshold:
                intentional_ratio = 0.7
            else:
                # Beyond threshold, decreasing proportion is intentional
                intentional_ratio = max(0.0, 0.7 - (total_drift - self.parameters.drift_detection_threshold))
            
            intentional_drift = total_drift * intentional_ratio
            unintentional_drift = total_drift - intentional_drift
        else:
            # Calculate intentional component (projection onto goal vector)
            goal_norm = np.linalg.norm(goal_vector)
            if goal_norm < 1e-10:
                intentional_drift = 0.0
            else:
                normalized_goal = goal_vector / goal_norm
                projection = np.dot(drift_vector, normalized_goal)
                intentional_drift = max(0.0, projection / goal_norm)
            
            # Calculate unintentional component (orthogonal to goal vector)
            unintentional_drift = max(0.0, total_drift - intentional_drift)
        
        return {
            "total": total_drift,
            "intentional": intentional_drift,
            "unintentional": unintentional_drift,
            "intentional_ratio": intentional_drift / total_drift if total_drift > 0 else 0.0,
        }
    
    def predict_drift(
        self,
        current_state: Dict[str, Any],
        inputs: List[Any],
        processing_function: Callable,
        n_steps: int = 5,
        reference_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict future drift based on current state and inputs.
        
        Args:
            current_state: Current system state
            inputs: Sequence of future inputs
            processing_function: Function that processes inputs to produce new states
            n_steps: Number of steps to predict
            reference_state: Reference state for drift calculation
            
        Returns:
            Dictionary with drift predictions
            
        Raises:
            ValueError: If inputs or processing_function are invalid
        """
        if not inputs:
            raise ValueError("Inputs cannot be empty")
        
        if not callable(processing_function):
            raise ValueError("Processing function must be callable")
        
        # Set reference state
        reference = reference_state or current_state.copy()
        
        # Initialize prediction state
        predicted_states = [current_state.copy()]
        predicted_drifts = [0.0]  # Initial drift from current state is 0
        
        # Predict future states and calculate drift
        state = current_state.copy()
        for i in range(min(n_steps, len(inputs))):
            try:
                # Generate next state
                next_state = processing_function(state, inputs[i])
                
                # Calculate drift from reference
                drift = self._calculate_state_distance(next_state, reference)
                
                # Store state and drift
                predicted_states.append(next_state)
                predicted_drifts.append(drift)
                
                # Update state for next iteration
                state = next_state
            except Exception as e:
                logger.error(f"Error in drift prediction at step {i}: {e}")
                break
        
        # Calculate drift velocity and acceleration
        incremental_drifts = [
            self._calculate_state_distance(predicted_states[i], predicted_states[i-1])
            for i in range(1, len(predicted_states))
        ]
        
        if len(incremental_drifts) >= 2:
            velocity = sum(incremental_drifts) / len(incremental_drifts)
            
            drift_changes = [
                incremental_drifts[i] - incremental_drifts[i-1]
                for i in range(1, len(incremental_drifts))
            ]
            
            acceleration = sum(drift_changes) / len(drift_changes) if drift_changes else 0.0
        else:
            velocity = incremental_drifts[0] if incremental_drifts else 0.0
            acceleration = 0.0
        
        # Determine predicted drift type
        drift_type = self._determine_drift_type(
            incremental_drifts, predicted_drifts[-1], reference
        )
        
        # Generate prediction analysis
        prediction_analysis = self._analyze_drift(
            predicted_drifts[-1], incremental_drifts, velocity, acceleration, drift_type
        )
        
        return {
            "predicted_states": predicted_states,
            "predicted_drifts": predicted_drifts,
            "final_drift": predicted_drifts[-1],
            "drift_velocity": velocity,
            "drift_acceleration": acceleration,
            "drift_type": drift_type,
            "analysis": prediction_analysis,
        }
    
    def stabilize_state(
        self,
        current_state: Dict[str, Any],
        reference_state: Dict[str, Any],
        stabilization_strength: float = 0.5,
        preserve_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Stabilize current state by pulling it toward reference state.
        
        Args:
            current_state: Current system state
            reference_state: Reference state to stabilize toward
            stabilization_strength: Strength of stabilization (0-1). Default 0.5.
            preserve_keys: List of keys to
