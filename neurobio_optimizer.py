"""
Neurobiologically-inspired optimizer for the Prompt Theory framework.

This module implements the neurobiological optimizer that extends the base optimizer
with cognitive science principles, creating prompts optimized based on human
neurobiological constraints and mechanisms.
"""

import copy
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import warnings
import time
import re

from prompt_theory.optimizers.base import PromptOptimizer
from prompt_theory.models.attention import AttentionModel
from prompt_theory.models.recursion import RecursiveProcessor
from prompt_theory.models.drift import DriftModel
from prompt_theory.utils.logging import get_logger
from prompt_theory.utils.validation import validate_parameters


logger = get_logger(__name__)


class NeurobiologicalOptimizer(PromptOptimizer):
    """Optimizer inspired by neurobiological principles.
    
    This class extends the base PromptOptimizer with specialized optimization
    techniques inspired by human cognitive processes, including working memory
    management, attention guidance, and recursive depth optimization.
    
    Attributes:
        cognitive_parameters: Dictionary of cognitive model parameters
        _cognitive_models: Dictionary of specialized cognitive models
    """
    
    def __init__(
        self,
        model: Union[str, Any],
        cognitive_parameters: Optional[Dict[str, Any]] = None,
        attention_model: Optional[AttentionModel] = None,
        recursive_processor: Optional[RecursiveProcessor] = None,
        drift_model: Optional[DriftModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize neurobiological optimizer.
        
        Args:
            model: LLM model identifier or client
            cognitive_parameters: Cognitive model parameters
            attention_model: Attention allocation model
            recursive_processor: Recursive processing model
            drift_model: Drift and stability model
            config: Additional configuration parameters
        """
        super().__init__(
            model=model,
            attention_model=attention_model,
            recursive_processor=recursive_processor,
            drift_model=drift_model,
            config=config,
        )
        
        # Initialize cognitive parameters with defaults if not provided
        self.cognitive_parameters = cognitive_parameters or {
            "working_memory": {
                "capacity": 7,  # Miller's 7±2
                "chunk_size": 4,  # Average chunk size
                "decay_rate": 0.1,  # Information decay rate
            },
            "attention": {
                "sustained_duration": 20,  # Seconds of sustained attention
                "context_switch_cost": 0.2,  # Cost of switching context (0-1)
                "salience_threshold": 0.3,  # Threshold for attention capture
            },
            "processing": {
                "cognitive_load_threshold": 0.7,  # Max cognitive load before performance drops
                "processing_speed_factor": 1.0,  # Relative processing speed
                "interference_sensitivity": 0.4,  # Sensitivity to interference
            },
            "memory": {
                "short_term_capacity": 4,  # Items in short-term memory
                "long_term_activation": 0.3,  # Activation threshold for LTM
                "elaboration_benefit": 0.5,  # Benefit of elaborative encoding
            },
            "metacognition": {
                "monitoring_accuracy": 0.7,  # Accuracy of metacognitive monitoring
                "control_effectiveness": 0.6,  # Effectiveness of metacognitive control
                "reflection_benefit": 0.4,  # Benefit of reflection
            },
        }
        
        # Initialize cognitive models
        self._cognitive_models = {
            "working_memory_model": self._initialize_working_memory_model(),
            "attention_guidance_model": self._initialize_attention_guidance_model(),
            "cognitive_load_model": self._initialize_cognitive_load_model(),
            "metacognition_model": self._initialize_metacognition_model(),
        }
        
        logger.debug(f"Initialized NeurobiologicalOptimizer with cognitive parameters")
    
    def _initialize_working_memory_model(self) -> Dict[str, Any]:
        """Initialize working memory model based on cognitive parameters."""
        wm_params = self.cognitive_parameters.get("working_memory", {})
        
        return {
            "capacity": wm_params.get("capacity", 7),
            "chunk_size": wm_params.get("chunk_size", 4),
            "decay_rate": wm_params.get("decay_rate", 0.1),
            "activation_function": lambda x: 1.0 / (1.0 + wm_params.get("decay_rate", 0.1) * x),
        }
    
    def _initialize_attention_guidance_model(self) -> Dict[str, Any]:
        """Initialize attention guidance model based on cognitive parameters."""
        attention_params = self.cognitive_parameters.get("attention", {})
        
        return {
            "sustained_duration": attention_params.get("sustained_duration", 20),
            "context_switch_cost": attention_params.get("context_switch_cost", 0.2),
            "salience_threshold": attention_params.get("salience_threshold", 0.3),
            "salience_weights": {
                "novelty": 0.3,
                "relevance": 0.4,
                "emotional_valence": 0.2,
                "complexity": 0.1,
            },
        }
    
    def _initialize_cognitive_load_model(self) -> Dict[str, Any]:
        """Initialize cognitive load model based on cognitive parameters."""
        processing_params = self.cognitive_parameters.get("processing", {})
        
        return {
            "load_threshold": processing_params.get("cognitive_load_threshold", 0.7),
            "processing_speed": processing_params.get("processing_speed_factor", 1.0),
            "interference_sensitivity": processing_params.get("interference_sensitivity", 0.4),
            "load_components": {
                "element_count": 0.3,
                "element_complexity": 0.3,
                "relationship_complexity": 0.2,
                "novelty": 0.2,
            },
        }
    
    def _initialize_metacognition_model(self) -> Dict[str, Any]:
        """Initialize metacognition model based on cognitive parameters."""
        metacog_params = self.cognitive_parameters.get("metacognition", {})
        
        return {
            "monitoring_accuracy": metacog_params.get("monitoring_accuracy", 0.7),
            "control_effectiveness": metacog_params.get("control_effectiveness", 0.6),
            "reflection_benefit": metacog_params.get("reflection_benefit", 0.4),
            "monitoring_components": {
                "comprehension": 0.4,
                "progress": 0.3,
                "confidence": 0.3,
            },
        }
    
    def optimize(
        self,
        base_prompt: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        cognitive_profile: Optional[Dict[str, Any]] = None,
        optimization_steps: Optional[List[str]] = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        trace_optimization: bool = False,
    ) -> str:
        """Optimize prompt using neurobiologically-inspired techniques.
        
        Args:
            base_prompt: Base prompt to optimize
            task: Task description
            context: Context information
            constraints: Optimization constraints
            cognitive_profile: Cognitive profile for audience-specific optimization
            optimization_steps: Specific optimization steps to apply
            max_iterations: Maximum optimization iterations. Default 5.
            convergence_threshold: Threshold for determining convergence. Default 0.01.
            trace_optimization: Whether to record detailed optimization trace. Default False.
            
        Returns:
            Neurobiologically optimized prompt
        """
        # Initialize context and constraints
        context = context or {}
        constraints = constraints or {}
        
        # Update cognitive parameters based on cognitive profile if provided
        if cognitive_profile:
            self._update_cognitive_parameters(cognitive_profile)
        
        # Determine optimization steps if not provided
        if optimization_steps is None:
            optimization_steps = self._determine_neurobiological_optimization_steps(
                task, context, constraints
            )
        
        # Call base optimizer's optimize method with neurobiological steps
        optimized_prompt = super().optimize(
            base_prompt=base_prompt,
            task=task,
            context=context,
            constraints=constraints,
            optimization_steps=optimization_steps,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            trace_optimization=trace_optimization,
        )
        
        return optimized_prompt
    
    def _update_cognitive_parameters(self, cognitive_profile: Dict[str, Any]) -> None:
        """Update cognitive parameters based on provided cognitive profile."""
        # Define cognitive profiles
        profiles = {
            "expert": {
                "working_memory": {"capacity": 9, "chunk_size": 5},
                "processing": {"cognitive_load_threshold": 0.8},
                "metacognition": {"monitoring_accuracy": 0.8},
            },
            "novice": {
                "working_memory": {"capacity": 5, "chunk_size": 3},
                "processing": {"cognitive_load_threshold": 0.6},
                "metacognition": {"monitoring_accuracy": 0.6},
            },
            "child": {
                "working_memory": {"capacity": 3, "chunk_size": 2},
                "processing": {"cognitive_load_threshold": 0.5},
                "metacognition": {"monitoring_accuracy": 0.4},
            },
            "elderly": {
                "working_memory": {"capacity": 5, "chunk_size": 3},
                "attention": {"sustained_duration": 15},
                "processing": {"processing_speed_factor": 0.8},
            },
            "adhd": {
                "attention": {"sustained_duration": 10, "context_switch_cost": 0.4},
                "processing": {"interference_sensitivity": 0.6},
            },
            "high_anxiety": {
                "attention": {"salience_threshold": 0.2},
                "processing": {"cognitive_load_threshold": 0.6},
                "metacognition": {"monitoring_accuracy": 0.6},
            },
        }
        
        # Apply profile-based updates
        if isinstance(cognitive_profile, str) and cognitive_profile in profiles:
            profile_params = profiles[cognitive_profile]
            self._apply_cognitive_profile(profile_params)
        elif isinstance(cognitive_profile, dict):
            self._apply_cognitive_profile(cognitive_profile)
    
    def _apply_cognitive_profile(self, profile_params: Dict[str, Dict[str, Any]]) -> None:
        """Apply cognitive profile parameters to current cognitive parameters."""
        for category, params in profile_params.items():
            if category in self.cognitive_parameters:
                self.cognitive_parameters[category].update(params)
        
        # Reinitialize cognitive models with updated parameters
        self._cognitive_models = {
            "working_memory_model": self._initialize_working_memory_model(),
            "attention_guidance_model": self._initialize_attention_guidance_model(),
            "cognitive_load_model": self._initialize_cognitive_load_model(),
            "metacognition_model": self._initialize_metacognition_model(),
        }
    
    def _determine_neurobiological_optimization_steps(
        self, task: str, context: Dict[str, Any], constraints: Dict[str, Any]
    ) -> List[str]:
        """Determine neurobiologically-inspired optimization steps based on task and context."""
        # Base optimization steps from neurobiological perspective
        neuro_steps = [
            "working_memory_optimization",
            "attention_guidance_optimization",
            "cognitive_load_management",
            "metacognitive_scaffolding",
            "chunking_optimization",
        ]
        
        # Task-specific adjustments
        task_specific_steps = {
            "reasoning": [
                "working_memory_optimization",
                "cognitive_load_management",
                "recursive_structure",
                "metacognitive_scaffolding",
                "attention_guidance_optimization",
            ],
            "creative": [
                "attention_guidance_optimization",
                "associative_priming",
                "cognitive_load_management",
                "emergence_enhancement",
                "working_memory_optimization",
            ],
            "educational": [
                "chunking_optimization",
                "working_memory_optimization",
                "metacognitive_scaffolding",
                "cognitive_load_management",
                "spaced_repetition_structure",
            ],
            "factual": [
                "working_memory_optimization",
                "semantic_organization",
                "chunking_optimization",
                "attention_guidance_optimization",
                "retrieval_optimization",
            ],
        }
        
        # Constraint-specific adjustments
        if "audience" in constraints:
            audience = constraints["audience"].lower()
            
            if audience in ["child", "novice", "beginner"]:
                return [
                    "working_memory_optimization",
                    "chunking_optimization",
                    "cognitive_load_management",
                    "attention_guidance_optimization",
                    "metacognitive_scaffolding",
                ]
            
            elif audience in ["expert", "advanced", "professional"]:
                return [
                    "semantic_organization",
                    "working_memory_optimization",
                    "metacognitive_scaffolding",
                    "recursive_structure",
                    "attention_guidance_optimization",
                ]
        
        # Use task-specific steps if available, otherwise default
        for task_keyword, steps in task_specific_steps.items():
            if task_keyword in task.lower():
                return steps
        
        return neuro_steps
    
    # Neurobiologically-inspired optimization methods
    
    def _optimize_working_memory_optimization(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt for working memory constraints.
        
        This method applies cognitive science principles about working memory
        to structure the prompt in a way that respects human working memory
        capacity and chunking mechanisms.
        """
        # Get working memory model parameters
        wm_model = self._cognitive_models["working_memory_model"]
        capacity = wm_model["capacity"]
        chunk_size = wm_model["chunk_size"]
        
        # Parse prompt into elements
        elements = self._parse_prompt_elements(prompt)
        
        # Count current working memory load
        current_load = len(elements)
        
        # If already within capacity, focus on chunking optimization
        if current_load <= capacity:
            return self._optimize_chunking(prompt, elements, capacity, chunk_size)
        
        # Otherwise, need to reduce elements to fit working memory
        logger.debug(f"Working memory optimization: reducing {current_load} elements to fit capacity {capacity}")
        
        # Organize elements by importance
        importance_scores = self._calculate_element_importance(elements, task, context)
        element_pairs = list(zip(elements, importance_scores))
        sorted_pairs = sorted(element_pairs, key=lambda x: x[1], reverse=True)
        
        # Keep most important elements within capacity
        kept_elements = [pair[0] for pair in sorted_pairs[:capacity]]
        
        # Check if we need to merge some lower-importance elements
        if len(sorted_pairs) > capacity:
            # Merge remaining elements into a single additional element
            remaining_elements = [pair[0] for pair in sorted_pairs[capacity:]]
            if remaining_elements:
                merged_element = self._merge_elements(remaining_elements)
                kept_elements.append(merged_element)
        
        # Apply chunking optimization to the reduced set
        optimized_prompt = self._optimize_chunking(
            self._assemble_prompt(kept_elements),
            kept_elements,
            capacity,
            chunk_size
        )
        
        return optimized_prompt
    
    def _optimize_chunking(
        self,
        prompt: str,
        elements: List[str],
        capacity: int,
        chunk_size: int,
    ) -> str:
        """Optimize prompt chunking to improve working memory efficiency."""
        # If few enough elements, focus on chunk organization
        if len(elements) <= capacity:
            # Apply organizational structure to create clear chunks
            structured_elements = []
            
            # Add organizational headers if needed
            if len(elements) >= 3 and not any(e.startswith("#") for e in elements):
                for i, element in enumerate(elements):
                    # Add headers to create explicit chunks
                    if i == 0:
                        # First element often serves as introduction
                        structured_elements.append(element)
                    else:
                        # Add a header based on content
                        header = self._generate_chunk_header(element)
                        structured_elements.append(f"## {header}\n\n{element}")
            else:
                structured_elements = elements
            
            return self._assemble_prompt(structured_elements)
        
        # If too many elements, need to create super-chunks
        chunks = []
        current_chunk = []
        
        for element in elements:
            current_chunk.append(element)
            if len(current_chunk) >= chunk_size:
                # Create a super-chunk with a header
                super_chunk = self._create_super_chunk(current_chunk)
                chunks.append(super_chunk)
                current_chunk = []
        
        # Add any remaining elements as a final chunk
        if current_chunk:
            super_chunk = self._create_super_chunk(current_chunk)
            chunks.append(super_chunk)
        
        return self._assemble_prompt(chunks)
    
    def _create_super_chunk(self, elements: List[str]) -> str:
        """Create a super-chunk from multiple elements with organizational structure."""
        # Generate a header that encapsulates the theme of these elements
        header = self._generate_chunk_header(" ".join(elements))
        
        # Format as a section with bullet points for sub-elements
        chunk_body = "\n".join(f"- {self._summarize_element(e)}" for e in elements)
        
        return f"## {header}\n\n{chunk_body}"
    
    def _generate_chunk_header(self, content: str) -> str:
        """Generate a descriptive header for a chunk based on its content."""
        # Extract key terms from content
        words = content.lower().split()
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", "and", "or"}
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Use most frequent terms to generate header
        if key_terms:
            from collections import Counter
            term_counts = Counter(key_terms)
            top_terms = [term for term, count in term_counts.most_common(3)]
            
            # Create header from top terms
            if len(top_terms) >= 2:
                return " ".join(t.capitalize() for t in top_terms[:2])
            else:
                return top_terms[0].capitalize()
        
        # Fallback
        return "Key Information"
    
    def _summarize_element(self, element: str) -> str:
        """Create a brief summary of an element for inclusion in a super-chunk."""
        # If element is short, use as is
        if len(element) < 100:
            return element
        
        # Extract first sentence
        sentences = element.split(".")
        if sentences:
            return sentences[0] + "..."
        
        # Fallback
        return element[:100] + "..."
    
    def _merge_elements(self, elements: List[str]) -> str:
        """Merge multiple elements into a single consolidated element."""
        # If few elements, just join them
        if len(elements) <= 3:
            return "\n\n".join(elements)
        
        # For many elements, create a summary section
        summary = "## Additional Information\n\n"
        for element in elements:
            # Extract a brief version of each element
            summary += f"- {self._summarize_element(element)}\n"
        
        return summary
    
    def _optimize_attention_guidance_optimization(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt with attention guidance mechanisms.
        
        This method applies cognitive science principles about attention
        to guide focus to key information and maintain engagement.
        """
        # Get attention model parameters
        attention_model = self._cognitive_models["attention_guidance_model"]
        salience_threshold = attention_model["salience_threshold"]
        salience_weights = attention_model["salience_weights"]
        
        # Parse prompt into elements
        elements = self._parse_prompt_elements(prompt)
        
        # Calculate salience for each element
        salience_scores = self._calculate_salience(elements, task, context, salience_weights)
        
        # Identify elements below salience threshold that need enhancement
        low_salience_indices = [
            i for i, score in enumerate(salience_scores) if score < salience_threshold
        ]
        
        if not low_salience_indices:
            # All elements have sufficient salience
            return prompt
        
        # Enhance low-salience elements
        enhanced_elements = elements.copy()
        for idx in low_salience_indices:
            enhanced_elements[idx] = self._enhance_salience(
                elements[idx], task, context
            )
        
        # Add visual attention guides if needed
        guided_elements = self._add_attention_guides(enhanced_elements, salience_scores)
        
        return self._assemble_prompt(guided_elements)
    
    def _calculate_salience(
        self,
        elements: List[str],
        task: str,
        context: Dict[str, Any],
        salience_weights: Dict[str, float],
    ) -> List[float]:
        """Calculate salience scores for each element based on attention factors."""
        salience_scores = []
        
        for element in elements:
            # Initialize component scores
            novelty = 0.0
            relevance = 0.0
            emotional_valence = 0.0
            complexity = 0.0
            
            # Calculate novelty (simplified)
            # Unusual or distinctive terms increase novelty
            unusual_terms = ["unique", "novel", "surprising", "unexpected", "new"]
            novelty = sum(term in element.lower() for term in unusual_terms) / 5
            
            # Calculate relevance to task and context
            task_terms = task.lower().split()
            context_terms = []
            for v in context.values():
                if isinstance(v, str):
                    context_terms.extend(v.lower().split())
            
            # Count matches
            task_matches = sum(term in element.lower() for term in task_terms)
            context_matches = sum(term in element.lower() for term in context_terms)
            
            # Normalize relevance score
            if task_terms:
                task_relevance = min(1.0, task_matches / len(task_terms))
            else:
                task_relevance = 0.5  # Neutral if no task terms
                
            if context_terms:
                context_relevance = min(1.0, context_matches / len(context_terms))
            else:
                context_relevance = 0.5  # Neutral if no context terms
                
            relevance = 0.7 * task_relevance + 0.3 * context_relevance
            
            # Calculate emotional valence (simplified)
            emotional_terms = [
                "important", "critical", "essential", "crucial", "key",
                "exciting", "interesting", "engaging", "surprising", 
                "warning", "caution", "note", "remember", "attention"
            ]
            emotional_valence = sum(term in element.lower() for term in emotional_terms) / len(emotional_terms)
            
            # Calculate complexity (simplified)
            words = element.split()
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            sentence_count = element.count(".") + element.count("!") + element.count("?")
            words_per_sentence = len(words) / max(1, sentence_count)
            
            # Normalize complexity (higher word length and words per sentence = higher complexity)
            complexity = min(1.0, (avg_word_length / 8) * (words_per_sentence / 20))
            
            # Calculate weighted salience score
            salience = (
                salience_weights["novelty"] * novelty +
                salience_weights["relevance"] * relevance +
                salience_weights["emotional_valence"] * emotional_valence +
                salience_weights["complexity"] * (1 - complexity)  # Invert complexity (simpler = more salient)
            )
            
            salience_scores.append(salience)
        
        return salience_scores
    
    def _enhance_salience(
        self,
        element: str,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """Enhance the salience of an element to improve attention capture."""
        # Strategies to enhance salience
        strategies = [
            self._add_emphasis,
            self._improve_relevance,
            self._enhance_structure,
            self._simplify_language,
        ]
        
        # Apply random strategy
        import random
        strategy = random.choice(strategies)
        
        return strategy(element, task, context)
    
    def _add_emphasis(
        self,
        element: str,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """Add emphasis markers to increase salience."""
        # If already has emphasis, return as is
        if "**" in element or "__" in element:
            return element
            
        # Add emphasis to first sentence if possible
        sentences = element.split(".")
        if len(sentences) > 1:
            # Emphasize first sentence
            enhanced = f"**{sentences[0].strip()}**." + ".".join(sentences[1:])
            return enhanced
            
        # For short elements, emphasize the whole thing
        if len(element) < 100:
            return f"**{element}**"
            
        # For longer elements, add a note
        return f"**Note:** {element}"
    
    def _improve_relevance(
        self,
        element: str,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """Improve relevance signaling to increase salience."""
        # Extract key terms from task
        task_terms = [term for term in task.lower().split() if len(term) > 3]
        
        # If element already contains task terms, add explicit relevance marker
        for term in task_terms:
            if term in element.lower():
                return f"[Relevant for {task}] {element}"
        
        # Otherwise, add general relevance marker
        return f"[Important] {element}"
    
    def _enhance_structure(
        self,
        element: str,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """Enhance structural organization to increase salience."""
        # If already structured, return as is
        if element.startswith("#") or "-" in element:
            return element
            
        # Convert to bullet points if long enough
        sentences = [s.strip() for s in element.split(".") if s.strip()]
        if len(sentences) >= 3:
            return "\n".join(f"- {s}." for s in sentences)
            
        # Add a header if there's content to put under it
        if len(element) > 100:
            header = self._generate_chunk_header(element)
            return f"### {header}\n\n{element}"
            
        # Default enhancement
        return element
    
    def _simplify_language(
        self,
        element: str,
        task: str,
        context: Dict[str, Any],
    ) -> str:
        """Simplify language to reduce complexity and increase salience."""
        # Identify complex words (simplified approach)
        complex_words = [
            "therefore", "subsequently", "consequently", "nevertheless",
            "furthermore", "additionally", "notwithstanding", "accordingly"
        ]
        
        # Replace complex words with simpler alternatives
        simplified = element
        replacements = {
            "therefore": "so",
            "subsequently": "then",
            "consequently": "as a result",
            "nevertheless": "still",
            "furthermore": "also",
            "additionally": "also",
            "notwithstanding": "despite this",
            "accordingly": "so"
        }
        
        for complex_word, simple_word in replacements.items():
            simplified = simplified.replace(complex_word, simple_word)
        
        return simplified
    
    def _add_attention_guides(
        self,
        elements: List[str],
        salience_scores: List[float],
    ) -> List[str]:
        """Add visual attention guides based on salience patterns."""
        # If fewer than 3 elements, no need for additional guides
        if len(elements) < 3:
            return elements
        
        # Identify high-salience elements (for anchoring attention)
        high_salience_indices = [
            i for i, score in enumerate(salience_scores) if score > 0.7
        ]
        
        # If no high-salience elements, create one
        if not high_salience_indices and elements:
            # Make the first element high salience
            elements[0] = f"# {elements[0]}" if not elements[0].startswith("#") else elements[0]
            high_salience_indices = [0]
        
        # Add guides based on element positions
        guided_elements = elements.copy()
        
        # Add explicit numbering if appropriate
        if len(elements) >= 4 and not any(e.startswith("#") for e in elements):
            # Number each element to create clearer structure
            for i in range(len(guided_elements)):
                if not guided_elements[i].startswith(("-", "#", "1.", "2.", "3.")):
                    guided_elements[i] = f"{i+1}. {guided_elements[i]}"
        
        return guided_elements
    
    def _optimize_cognitive_load_management(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt to manage cognitive load.
        
        This method applies cognitive science principles to ensure the prompt
        doesn't exceed cognitive load thresholds, allowing for optimal processing.
        """
        # Get cognitive load model parameters
        load_model = self._cognitive_models["cognitive_load_model"]
        load_threshold = load_model["load_threshold"]
        load_components = load_model["load_components"]
        
        # Break prompt into elements
        elements = self._parse_prompt_elements(prompt)
        
        # Calculate cognitive load
        current_load = self._calculate_cognitive_load(elements, load_components)
        
        # If below threshold, no optimization needed
        if current_load <= load_threshold:
            return prompt
        
        # Apply cognitive load reduction techniques
        logger.debug(f"Cognitive load optimization: reducing load from {current_load:.2f} to below {load_threshold:.2f}")
        
        # Determine which technique to apply base
