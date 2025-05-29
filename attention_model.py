"""
Attention allocation model for the Prompt Theory framework.

This module implements the mathematical models for attention allocation in both AI and
human cognitive systems, providing a unified framework for understanding and optimizing
attentional processes across domains.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from prompt_theory.utils.math import softmax, cosine_similarity
from prompt_theory.utils.logging import get_logger
from prompt_theory.utils.validation import validate_parameters


logger = get_logger(__name__)


@dataclass
class AttentionParameters:
    """Parameters controlling attention allocation behavior."""
    
    capacity: float = 7.0  # Working memory-inspired capacity limit
    recency_bias: float = 0.3  # Weight for recency effects (0-1)
    salience_weight: float = 0.5  # Weight for salience factors (0-1)
    attention_decay: float = 0.1  # Attention decay rate over context elements
    cross_modal_weight: float = 0.6  # Weight for cross-modal attention (0-1)
    depth_sensitivity: float = 0.2  # Sensitivity to recursive depth


class AttentionModel:
    """Models attention allocation in both AI and human systems.
    
    This class implements the core attention allocation mechanisms defined in the
    Prompt Theory mathematical framework, providing a unified approach to modeling
    attentional processes across AI and human cognitive systems.
    
    Attributes:
        parameters: AttentionParameters controlling attention behavior
        _state: Internal state tracking attention allocation history
    """
    
    def __init__(
        self,
        capacity: Optional[float] = None,
        recency_bias: float = 0.3,
        salience_weight: float = 0.5,
        attention_decay: float = 0.1,
        cross_modal_weight: float = 0.6,
        depth_sensitivity: float = 0.2,
    ):
        """Initialize attention model parameters.
        
        Args:
            capacity: Attention capacity limit. Defaults to system-specific value.
            recency_bias: Weight for recency bias (0-1). Default 0.3.
            salience_weight: Weight for salience factors (0-1). Default 0.5.
            attention_decay: Attention decay rate over context elements. Default 0.1.
            cross_modal_weight: Weight for cross-modal attention (0-1). Default 0.6.
            depth_sensitivity: Sensitivity to recursive depth. Default 0.2.
        
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        validate_parameters(
            ("recency_bias", recency_bias, 0.0, 1.0),
            ("salience_weight", salience_weight, 0.0, 1.0),
            ("attention_decay", attention_decay, 0.0, 1.0),
            ("cross_modal_weight", cross_modal_weight, 0.0, 1.0),
            ("depth_sensitivity", depth_sensitivity, 0.0, 1.0),
        )
        
        self.parameters = AttentionParameters(
            capacity=capacity if capacity is not None else 7.0,  # Default based on human working memory
            recency_bias=recency_bias,
            salience_weight=salience_weight,
            attention_decay=attention_decay,
            cross_modal_weight=cross_modal_weight,
            depth_sensitivity=depth_sensitivity,
        )
        
        self._state = {
            "attention_history": [],
            "current_allocation": {},
            "cognitive_load": 0.0,
        }
        
        logger.debug(f"Initialized AttentionModel with parameters: {self.parameters}")
    
    def allocate(
        self,
        context: List[Any],
        queries: Optional[List[Any]] = None,
        keys: Optional[List[Any]] = None,
        values: Optional[List[Any]] = None,
        salience_factors: Optional[Dict[int, float]] = None,
        recency_indices: Optional[List[int]] = None,
        recursion_depth: int = 0,
    ) -> Dict[int, float]:
        """Allocate attention across context elements.
        
        This method implements the core attention allocation mechanism defined in
        Equation (5) of the Prompt Theory mathematical framework:
        
        A(X) = softmax(Q(X) · K(X)^T / √d)
        
        With modifications for recency and salience as defined in Equation (6):
        
        A'(X) = λ · A(X) + (1-λ) · R(X)
        
        Args:
            context: List of context elements to allocate attention across
            queries: List of query vectors (if using transformer-style attention)
            keys: List of key vectors (if using transformer-style attention)
            values: List of value vectors (if using transformer-style attention)
            salience_factors: Dictionary mapping context indices to salience scores
            recency_indices: List of indices in order of recency (most recent first)
            recursion_depth: Current recursion depth affecting attention
            
        Returns:
            Dictionary mapping context indices to attention allocation weights
        
        Raises:
            ValueError: If inputs are inconsistent or invalid
        """
        if not context:
            raise ValueError("Context cannot be empty")
        
        context_size = len(context)
        
        # Default queries, keys, values if not provided
        if queries is None:
            queries = context
        if keys is None:
            keys = context
        if values is None:
            values = context
            
        # Default salience and recency if not provided
        if salience_factors is None:
            salience_factors = {i: 1.0 for i in range(context_size)}
        if recency_indices is None:
            recency_indices = list(range(context_size))[::-1]  # Most recent last
            
        # Validate inputs
        if len(queries) != context_size or len(keys) != context_size or len(values) != context_size:
            raise ValueError("Queries, keys, and values must have same length as context")
            
        # Step 1: Calculate base attention using transformer-style QK product
        attention_base = self._calculate_base_attention(queries, keys, context_size)
        
        # Step 2: Calculate recency-based attention
        attention_recency = self._calculate_recency_attention(recency_indices, context_size)
        
        # Step 3: Calculate salience-based attention
        attention_salience = self._calculate_salience_attention(salience_factors, context_size)
        
        # Step 4: Combine attention mechanisms with weighting
        lambda_val = self.parameters.salience_weight
        attention_combined = {}
        
        for i in range(context_size):
            # Apply recency and salience weights as per Equation (6)
            combined = (
                (1 - self.parameters.recency_bias) * attention_base[i] +
                self.parameters.recency_bias * attention_recency[i]
            )
            
            # Apply salience weighting
            combined = (
                (1 - lambda_val) * combined +
                lambda_val * attention_salience[i]
            )
            
            # Apply recursion depth effects
            depth_factor = 1.0 / (1.0 + self.parameters.depth_sensitivity * recursion_depth)
            combined *= depth_factor
            
            attention_combined[i] = combined
        
        # Step 5: Normalize and apply capacity constraint
        attention_normalized = self._normalize_and_constrain(attention_combined)
        
        # Update internal state
        self._state["attention_history"].append(attention_normalized)
        self._state["current_allocation"] = attention_normalized
        self._state["cognitive_load"] = self._calculate_cognitive_load(attention_normalized)
        
        return attention_normalized
    
    def _calculate_base_attention(
        self, queries: List[Any], keys: List[Any], context_size: int
    ) -> Dict[int, float]:
        """Calculate base attention using transformer-style QK product."""
        attention = {}
        
        # Handle different input types for queries/keys
        if isinstance(queries[0], (np.ndarray, list)) and isinstance(keys[0], (np.ndarray, list)):
            # Vector inputs - use dot product
            for i in range(context_size):
                query_vec = np.array(queries[i])
                attention[i] = sum(
                    cosine_similarity(query_vec, np.array(keys[j]))
                    for j in range(context_size)
                )
        else:
            # Non-vector inputs - use generic similarity
            for i in range(context_size):
                attention[i] = 1.0  # Equal base attention
        
        return attention
    
    def _calculate_recency_attention(
        self, recency_indices: List[int], context_size: int
    ) -> Dict[int, float]:
        """Calculate attention based on recency effects."""
        attention = {}
        
        # Recency weighting: more recent items get more attention
        for i in range(context_size):
            # Find position in recency ordering (0 = most recent)
            recency_position = recency_indices.index(i)
            
            # Apply exponential decay based on recency
            attention[i] = np.exp(-self.parameters.attention_decay * recency_position)
            
        return attention
    
    def _calculate_salience_attention(
        self, salience_factors: Dict[int, float], context_size: int
    ) -> Dict[int, float]:
        """Calculate attention based on salience factors."""
        attention = {}
        
        for i in range(context_size):
            attention[i] = salience_factors.get(i, 1.0)
            
        return attention
    
    def _normalize_and_constrain(self, attention: Dict[int, float]) -> Dict[int, float]:
        """Normalize attention weights and apply capacity constraints."""
        # Extract values and indices
        indices = list(attention.keys())
        values = [attention[i] for i in indices]
        
        # Apply softmax for basic normalization
        normalized_values = softmax(values)
        
        # Apply capacity constraint - if too many items, suppress low-attention items
        if len(indices) > self.parameters.capacity:
            # Sort by attention value
            sorted_pairs = sorted(zip(indices, normalized_values), key=lambda x: x[1], reverse=True)
            
            # Keep top items at full strength, suppress others
            for idx, (i, v) in enumerate(sorted_pairs):
                if idx >= self.parameters.capacity:
                    # Apply exponential suppression beyond capacity
                    suppression = np.exp(-(idx - self.parameters.capacity + 1))
                    normalized_values[indices.index(i)] *= suppression
            
            # Re-normalize after suppression
            sum_values = sum(normalized_values)
            if sum_values > 0:
                normalized_values = [v / sum_values for v in normalized_values]
        
        # Rebuild dictionary with normalized values
        return {indices[i]: normalized_values[i] for i in range(len(indices))}
    
    def _calculate_cognitive_load(self, attention: Dict[int, float]) -> float:
        """Calculate cognitive load based on attention distribution."""
        # Cognitive load increases with:
        # 1. Number of items attended to
        # 2. Evenness of attention distribution (more divided attention = higher load)
        
        attention_values = list(attention.values())
        
        # Measure of how many items are being actively attended to
        effective_items = sum(v > 0.1 for v in attention_values)  # Items with significant attention
        
        # Entropy of attention distribution (higher entropy = more divided attention)
        non_zero_values = [v for v in attention_values if v > 0]
        entropy = -sum(v * np.log(v) for v in non_zero_values) if non_zero_values else 0
        
        # Combine factors to estimate cognitive load
        load = (effective_items / self.parameters.capacity) + (entropy / np.log(len(attention)))
        
        return min(1.0, load)  # Normalize to 0-1 scale
    
    def predict_human_attention(
        self,
        stimuli: List[Any],
        task: str,
        individual_params: Optional[Dict[str, float]] = None,
    ) -> Dict[int, float]:
        """Predict human attention allocation for given stimuli.
        
        Args:
            stimuli: List of stimuli to predict attention allocation for
            task: Task description
            individual_params: Individual-specific parameters
            
        Returns:
            Dictionary mapping stimuli indices to predicted attention
        """
        # Adjust model parameters based on individual differences if provided
        original_params = self.parameters
        if individual_params:
            # Temporarily update parameters based on individual differences
            for param_name, value in individual_params.items():
                if hasattr(self.parameters, param_name):
                    setattr(self.parameters, param_name, value)
        
        # Task-specific parameter adjustments
        task_adjustments = {
            "visual_search": {"salience_weight": 0.7, "recency_bias": 0.2},
            "reading": {"salience_weight": 0.4, "recency_bias": 0.5},
            "problem_solving": {"salience_weight": 0.6, "recency_bias": 0.3},
            "memorization": {"salience_weight": 0.5, "recency_bias": 0.6},
        }
        
        # Apply task-specific adjustments if available
        task_params = task_adjustments.get(task.lower(), {})
        for param_name, value in task_params.items():
            if hasattr(self.parameters, param_name):
                setattr(self.parameters, param_name, value)
        
        # Estimate salience based on stimulus characteristics
        # This is a simplified approach; real implementation would use more sophisticated
        # feature extraction and salience estimation
        salience_factors = self._estimate_salience(stimuli, task)
        
        # Allocate attention using the core mechanism
        attention = self.allocate(
            context=stimuli,
            salience_factors=salience_factors,
        )
        
        # Restore original parameters
        self.parameters = original_params
        
        return attention
    
    def _estimate_salience(self, stimuli: List[Any], task: str) -> Dict[int, float]:
        """Estimate salience factors for stimuli based on characteristics and task."""
        salience = {}
        
        # This is a placeholder implementation
        # A real implementation would extract features from stimuli and
        # calculate salience based on task-relevant characteristics
        
        for i, stimulus in enumerate(stimuli):
            # Default salience
            salience[i] = 1.0
            
            # Example feature: length/complexity affects salience
            if hasattr(stimulus, "__len__"):
                complexity = min(len(stimulus) / 100, 2.0)
                salience[i] *= (1.0 + complexity * 0.5)
            
            # Example feature: novelty/distinctiveness
            # (simplified here as random variation)
            distinctiveness = 0.8 + 0.4 * np.random.random()
            salience[i] *= distinctiveness
        
        return salience
    
    def optimize_for_focus(
        self,
        context: List[Any],
        target_elements: List[int],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Optimize context to focus attention on target elements.
        
        Args:
            context: Current context elements
            target_elements: Elements to focus attention on
            constraints: Optimization constraints
            
        Returns:
            Optimized context to enhance focus on target elements
        """
        if not context or not target_elements:
            return context.copy() if context else []
        
        constraints = constraints or {}
        
        # Start with a copy of the original context
        optimized_context = context.copy()
        
        # Predict current attention allocation
        current_attention = self.allocate(context=context)
        
        # Calculate attention on target elements
        target_attention = sum(current_attention.get(i, 0) for i in target_elements)
        logger.debug(f"Initial target attention: {target_attention:.4f}")
        
        # Apply optimization techniques based on constraints
        techniques = [
            self._apply_salience_enhancement,
            self._apply_position_optimization,
            self._apply_distractor_reduction,
        ]
        
        for technique in techniques:
            optimized_context = technique(
                optimized_context, target_elements, constraints
            )
            
            # Check if optimization improved target attention
            new_attention = self.allocate(context=optimized_context)
            new_target_attention = sum(new_attention.get(i, 0) for i in target_elements)
            
            logger.debug(f"After {technique.__name__}: target attention = {new_target_attention:.4f}")
            
            # If we've achieved high attention on targets, we can stop
            if new_target_attention > 0.8:
                break
        
        return optimized_context
    
    def _apply_salience_enhancement(
        self, context: List[Any], target_elements: List[int], constraints: Dict[str, Any]
    ) -> List[Any]:
        """Enhance salience of target elements."""
        # This is a simplified implementation
        # A real implementation would modify the content to increase salience
        # through formatting, emphasis, or content modification
        return context
    
    def _apply_position_optimization(
        self, context: List[Any], target_elements: List[int], constraints: Dict[str, Any]
    ) -> List[Any]:
        """Optimize positions of elements to enhance attention on targets."""
        # This is a simplified implementation
        # A real implementation would reorder elements based on recency effects
        # and structural importance
        
        # Simple approach: move target elements to positions of high attention
        # (beginning and end due to primacy and recency effects)
        optimized = context.copy()
        
        # If allowed by constraints, reorder to position targets at attention hotspots
        if constraints.get("allow_reordering", True):
            # Extract target elements
            targets = [optimized[i] for i in target_elements if i < len(optimized)]
            non_targets = [item for i, item in enumerate(optimized) if i not in target_elements]
            
            # Reorder with targets at beginning and end
            if len(targets) == 1:
                # Single target - put at beginning
                optimized = targets + non_targets
            elif len(targets) == 2:
                # Two targets - put at beginning and end
                optimized = targets[:1] + non_targets + targets[1:]
            else:
                # Multiple targets - distribute at beginning, middle and end
                third = max(1, len(targets) // 3)
                optimized = targets[:third] + non_targets[:len(non_targets)//2] + \
                            targets[third:2*third] + non_targets[len(non_targets)//2:] + \
                            targets[2*third:]
        
        return optimized
    
    def _apply_distractor_reduction(
        self, context: List[Any], target_elements: List[int], constraints: Dict[str, Any]
    ) -> List[Any]:
        """Reduce influence of distractor elements."""
        # This is a simplified implementation
        # A real implementation would modify or remove distracting elements
        # that compete for attention with target elements
        return context
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the attention model."""
        return self._state.copy()
    
    def reset_state(self) -> None:
        """Reset the internal state of the attention model."""
        self._state = {
            "attention_history": [],
            "current_allocation": {},
            "cognitive_load": 0.0,
        }


class MultiModalAttentionModel(AttentionModel):
    """Extension of AttentionModel for multi-modal attention allocation.
    
    This class extends the base attention model to handle cross-modal attention
    effects, such as interactions between text, images, and other modalities.
    """
    
    def __init__(
        self,
        modalities: List[str],
        cross_modal_attention_weight: float = 0.6,
        modal_specific_capacities: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Initialize multi-modal attention model.
        
        Args:
            modalities: List of modalities to support
            cross_modal_attention_weight: Weight for cross-modal attention effects
            modal_specific_capacities: Dictionary mapping modalities to capacity values
            **kwargs: Additional parameters for base AttentionModel
        """
        super().__init__(**kwargs)
        
        self.modalities = modalities
        self.cross_modal_weight = cross_modal_attention_weight
        
        # Set up modality-specific capacities
        self.modal_capacities = modal_specific_capacities or {}
        for modality in modalities:
            if modality not in self.modal_capacities:
                # Default capacities based on cognitive research
                default_capacities = {
                    "text": 7.0,  # Classic Miller's 7±2 for verbal
                    "image": 4.0,  # Lower capacity for visual objects
                    "code": 5.0,  # Programming constructs
                    "audio": 4.0,  # Auditory objects
                    "video": 3.0,  # Dynamic visual scenes
                }
                self.modal_capacities[modality] = default_capacities.get(modality, 5.0)
        
        # Extended state for multi-modal tracking
        self._state["modal_allocations"] = {modality: {} for modality in modalities}
        
        logger.debug(f"Initialized MultiModalAttentionModel with modalities: {modalities}")
    
    def allocate_multimodal(
        self,
        contexts: Dict[str, List[Any]],
        queries: Optional[Dict[str, List[Any]]] = None,
        keys: Optional[Dict[str, List[Any]]] = None,
        values: Optional[Dict[str, List[Any]]] = None,
        salience_factors: Optional[Dict[str, Dict[int, float]]] = None,
        recency_indices: Optional[Dict[str, List[int]]] = None,
        cross_modal_interactions: Optional[Dict[Tuple[str, int, str, int], float]] = None,
    ) -> Dict[str, Dict[int, float]]:
        """Allocate attention across multiple modalities.
        
        Args:
            contexts: Dictionary mapping modalities to context elements
            queries: Dictionary mapping modalities to query vectors
            keys: Dictionary mapping modalities to key vectors
            values: Dictionary mapping modalities to value vectors
            salience_factors: Dictionary mapping modalities to salience dictionaries
            recency_indices: Dictionary mapping modalities to recency indices
            cross_modal_interactions: Dictionary mapping (modality1, idx1, modality2, idx2)
                tuples to interaction strengths
        
        Returns:
            Dictionary mapping modalities to attention allocation dictionaries
        """
        # Initialize default dictionaries if not provided
        queries = queries or {}
        keys = keys or {}
        values = values or {}
        salience_factors = salience_factors or {}
        recency_indices = recency_indices or {}
        cross_modal_interactions = cross_modal_interactions or {}
        
        # Step 1: Calculate base attention for each modality independently
        base_allocations = {}
        for modality in self.modalities:
            if modality in contexts:
                # Save original capacity
                original_capacity = self.parameters.capacity
                
                # Set modality-specific capacity
                self.parameters.capacity = self.modal_capacities.get(modality, original_capacity)
                
                # Allocate attention for this modality
                base_allocations[modality] = self.allocate(
                    context=contexts[modality],
                    queries=queries.get(modality),
                    keys=keys.get(modality),
                    values=values.get(modality),
                    salience_factors=salience_factors.get(modality),
                    recency_indices=recency_indices.get(modality),
                )
                
                # Restore original capacity
                self.parameters.capacity = original_capacity
        
        # Step 2: Apply cross-modal interactions
        if cross_modal_interactions:
            self._apply_cross_modal_effects(base_allocations, cross_modal_interactions)
        
        # Update state with modal allocations
        self._state["modal_allocations"] = base_allocations
        
        return base_allocations
    
    def _apply_cross_modal_effects(
        self,
        allocations: Dict[str, Dict[int, float]],
        interactions: Dict[Tuple[str, int, str, int], float],
    ) -> None:
        """Apply cross-modal interaction effects to attention allocations."""
        # This modifies allocations in-place
        
        # For each interaction (modality1, idx1, modality2, idx2) -> strength
        for (mod1, idx1, mod2, idx2), strength in interactions.items():
            if mod1 in allocations and mod2 in allocations:
                if idx1 in allocations[mod1] and idx2 in allocations[mod2]:
                    # Calculate influence based on interaction strength and current attention
                    influence1to2 = allocations[mod1][idx1] * strength * self.cross_modal_weight
                    influence2to1 = allocations[mod2][idx2] * strength * self.cross_modal_weight
                    
                    # Apply influences
                    allocations[mod2][idx2] += influence1to2
                    allocations[mod1][idx1] += influence2to1
        
        # Renormalize each modality's allocation
        for modality, allocation in allocations.items():
            values = list(allocation.values())
            normalized = softmax(values)
            for i, idx in enumerate(allocation.keys()):
                allocation[idx] = normalized[i]


# Additional classes and functions for specialized attention modeling

class AttentionAnalyzer:
    """Analyzes attention patterns in model outputs or human behavior."""
    
    def __init__(self, attention_model: AttentionModel):
        """Initialize with an attention model."""
        self.attention_model = attention_model
    
    def analyze_attention_pattern(
        self, 
        attention_allocation: Dict[int, float]
    ) -> Dict[str, Any]:
        """Analyze an attention allocation pattern for key characteristics."""
        # Extract values
        indices = sorted(attention_allocation.keys())
        values = [attention_allocation[i] for i in indices]
        
        # Calculate key metrics
        metrics = {
            "focus_level": self._calculate_focus_level(values),
            "attention_entropy": self._calculate_attention_entropy(values),
            "attended_items": self._calculate_effective_items(values),
            "peak_indices": self._find_attention_peaks(attention_allocation),
            "neglected_indices": self._find_neglected_items(attention_allocation),
        }
        
        return metrics
    
    def _calculate_focus_level(self, attention_values: List[float]) -> float:
        """Calculate level of focus vs. distribution in attention."""
        # Gini coefficient as a measure of attention inequality
        # Higher values indicate more focused attention
        sorted_values = sorted(attention_values)
        n = len(sorted_values)
        if n <= 1 or sum(sorted_values) == 0:
            return 0.0
            
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] * n)) / n
    
    def _calculate_attention_entropy(self, attention_values: List[float]) -> float:
        """Calculate entropy of attention distribution."""
        # Normalize if needed
        sum_values = sum(attention_values)
        if sum_values == 0:
            return 0.0
            
        normalized = [v / sum_values for v in attention_values if v > 0]
        
        # Calculate entropy
        return -sum(p * np.log2(p) for p in normalized)
    
    def _calculate_effective_items(self, attention_values: List[float]) -> float:
        """Calculate effective number of attended items."""
        # Items with significant attention (>0.1)
        return sum(1 for v in attention_values if v > 0.1)
    
    def _find_attention_peaks(self, attention_allocation: Dict[int, float]) -> List[int]:
        """Find indices with peak attention."""
        threshold = 0.7 * max(attention_allocation.values())
        return [i for i, v in attention_allocation.items() if v >= threshold]
    
    def _find_neglected_items(self, attention_allocation: Dict[int, float]) -> List[int]:
        """Find indices with negligible attention."""
        threshold = 0.05
        return [i for i, v in attention_allocation.items() if v <= threshold]
    
    def compare_attention_patterns(
        self, 
        pattern1: Dict[int, float], 
        pattern2: Dict[int, float]
    ) -> Dict[str, float]:
        """Compare two attention allocation patterns."""
        # Ensure patterns have same indices
        all_indices = set(pattern1.keys()).union(set(pattern2.keys()))
        p1 = {i: pattern1.get(i, 0.0) for i in all_indices}
        p2 = {i: pattern2.get(i, 0.0) for i in all_indices}
        
        # Extract values
        indices = sorted(all_indices)
        values1 = [p1[i] for i in indices]
        values2 = [p2[i] for i in indices]
        
        # Calculate comparison metrics
        metrics = {
            "cosine_similarity": self._cosine_similarity(values1, values2),
            "peak_alignment": self._calculate_peak_alignment(p1, p2),
            "distributional_difference": self._jensen_shannon_divergence(values1, values2),
            "focus_difference": abs(self._calculate_focus_level(values1) - 
                                   self._calculate_focus_level(values2)),
        }
        
        return metrics
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
            
        norm1 = np.sqrt(sum(x*x for x in v1))
        norm2 = np.sqrt(sum(x*x for x in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return sum(x*y for x, y in zip(v1, v2)) / (norm1 * norm2)
    
    def _calculate_peak_alignment(
        self, pattern1: Dict[int, float], pattern2: Dict[int, float]
    ) -> float:
        """Calculate alignment of attention peaks between patterns."""
        peaks1 = self._find_attention_peaks(pattern1)
        peaks2 = self._find_attention_peaks(pattern2)
        
        if not peaks1 or not peaks2:
            return 0.0
            
        # Jaccard similarity of peak sets
        intersection = len(set(peaks1).intersection(set(peaks2)))
        union = len(set(peaks1).union(set(peaks2)))
        
        return intersection / union if union > 0 else 0.0
