"""
Base optimizer for the Prompt Theory framework.

This module implements the base prompt optimization class that integrates attention,
recursion, and drift models to create optimized prompts for both AI and human
cognitive systems.
"""

import copy
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import warnings
import time
import json

from prompt_theory.models.attention import AttentionModel
from prompt_theory.models.recursion import RecursiveProcessor
from prompt_theory.models.drift import DriftModel
from prompt_theory.utils.logging import get_logger
from prompt_theory.utils.validation import validate_parameters


logger = get_logger(__name__)


class PromptOptimizer:
    """Base class for prompt optimization.
    
    This class integrates the core Prompt Theory models (attention, recursion, and drift)
    to optimize prompts for both AI and human cognitive systems. It serves as the
    foundation for specialized optimizers like NeurobiologicalOptimizer.
    
    Attributes:
        model: LLM model identifier or client
        attention_model: Model for attention allocation
        recursive_processor: Model for recursive processing
        drift_model: Model for drift and stability
        _state: Internal state tracking optimization history
    """
    
    def __init__(
        self,
        model: Union[str, Any],
        attention_model: Optional[AttentionModel] = None,
        recursive_processor: Optional[RecursiveProcessor] = None,
        drift_model: Optional[DriftModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize prompt optimizer.
        
        Args:
            model: LLM model identifier or client
            attention_model: Attention allocation model
            recursive_processor: Recursive processing model
            drift_model: Drift and stability model
            config: Additional configuration parameters
            
        Raises:
            ValueError: If model is invalid
        """
        self.config = config or {}
        
        # Initialize model
        self.model = self._initialize_model(model)
        
        # Initialize component models with defaults if not provided
        self.attention_model = attention_model or AttentionModel(
            capacity=self.config.get("attention_capacity", 7),
            recency_bias=self.config.get("recency_bias", 0.3),
            salience_weight=self.config.get("salience_weight", 0.5),
        )
        
        self.recursive_processor = recursive_processor or RecursiveProcessor(
            max_depth=self.config.get("max_recursion_depth", 4),
            collapse_threshold=self.config.get("collapse_threshold", 0.8),
            emergence_threshold=self.config.get("emergence_threshold", 0.6),
        )
        
        self.drift_model = drift_model or DriftModel(
            stability_params=self.config.get("stability_params", None),
            drift_detection_threshold=self.config.get("drift_threshold", 0.3),
        )
        
        # Initialize state
        self._state = {
            "optimization_history": [],
            "prompt_versions": [],
            "effectiveness_scores": {},
            "current_prompt": None,
            "reference_prompt": None,
        }
        
        logger.debug(f"Initialized PromptOptimizer with model: {model}")
    
    def _initialize_model(self, model: Union[str, Any]) -> Any:
        """Initialize LLM model based on input type.
        
        Args:
            model: Model identifier (string) or model client (object)
            
        Returns:
            Initialized model client
            
        Raises:
            ValueError: If model initialization fails
        """
        # If model is already a client object, return it directly
        if not isinstance(model, str):
            return model
        
        # Initialize model based on string identifier
        if model.startswith("gpt-"):
            try:
                import openai
                return openai.OpenAI()
            except ImportError:
                raise ValueError(
                    "OpenAI Python package not installed. "
                    "Install it with: pip install openai"
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI client: {e}")
        
        elif model.startswith("claude-"):
            try:
                import anthropic
                return anthropic.Anthropic()
            except ImportError:
                raise ValueError(
                    "Anthropic Python package not installed. "
                    "Install it with: pip install anthropic"
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Anthropic client: {e}")
        
        elif model.startswith("llama-"):
            try:
                from prompt_theory.utils.llm_clients import LlamaClient
                return LlamaClient(model_name=model)
            except ImportError:
                raise ValueError(
                    "Llama client dependencies not installed. "
                    "Install them with: pip install prompt-theory[llama]"
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Llama client: {e}")
        
        else:
            # Generic model initialization
            try:
                from prompt_theory.utils.llm_clients import GenericModelClient
                return GenericModelClient(model_name=model)
            except ImportError:
                raise ValueError(
                    "Model client dependencies not installed. "
                    "Install them with: pip install prompt-theory[models]"
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize model client: {e}")
    
    def optimize(
        self,
        base_prompt: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        optimization_steps: Optional[List[str]] = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        trace_optimization: bool = False,
    ) -> str:
        """Optimize prompt for given task and context.
        
        This method applies the Prompt Theory optimization framework to improve a prompt
        by analyzing attention allocation, recursive processing, and drift/stability.
        
        Args:
            base_prompt: Base prompt to optimize
            task: Task description
            context: Context information
            constraints: Optimization constraints
            optimization_steps: Specific optimization steps to apply
            max_iterations: Maximum optimization iterations. Default 5.
            convergence_threshold: Threshold for determining convergence. Default 0.01.
            trace_optimization: Whether to record detailed optimization trace. Default False.
            
        Returns:
            Optimized prompt
            
        Raises:
            ValueError: If base_prompt is empty or invalid
        """
        if not base_prompt or not isinstance(base_prompt, str):
            raise ValueError("Base prompt must be a non-empty string")
        
        # Initialize context and constraints
        context = context or {}
        constraints = constraints or {}
        
        # Determine optimization steps
        if optimization_steps is None:
            optimization_steps = self._determine_optimization_steps(task, context, constraints)
        
        # Initialize optimization state
        current_prompt = base_prompt
        prompt_versions = [current_prompt]
        effectiveness_scores = []
        
        # Initialize trace if requested
        trace = [] if trace_optimization else None
        
        # Store reference prompt
        self._state["reference_prompt"] = base_prompt
        
        # Main optimization loop
        for iteration in range(max_iterations):
            logger.info(f"Starting optimization iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate current prompt
            current_score = self.evaluate(current_prompt, task, context)
            effectiveness_scores.append(current_score)
            
            logger.debug(f"Current effectiveness score: {current_score:.4f}")
            
            # Check for convergence
            if iteration > 0:
                improvement = current_score - effectiveness_scores[-2]
                if abs(improvement) < convergence_threshold:
                    logger.info(f"Optimization converged after {iteration + 1} iterations")
                    break
            
            # Apply optimization steps
            try:
                new_prompt = self._apply_optimization_steps(
                    current_prompt, optimization_steps, task, context, constraints
                )
                
                # Calculate prompt change
                prompt_change = self._calculate_prompt_change(current_prompt, new_prompt)
                
                # Record trace if requested
                if trace_optimization:
                    trace.append({
                        "iteration": iteration + 1,
                        "current_prompt": current_prompt,
                        "current_score": current_score,
                        "new_prompt": new_prompt,
                        "prompt_change": prompt_change,
                        "optimization_steps": optimization_steps,
                    })
                
                # Update current prompt
                current_prompt = new_prompt
                prompt_versions.append(current_prompt)
                
                # Check if we've reached maximum improvement
                if current_score >= 0.95:
                    logger.info("Reached excellent effectiveness score, stopping optimization")
                    break
                
            except Exception as e:
                logger.error(f"Error in optimization iteration {iteration + 1}: {e}")
                break
        
        # Evaluate final prompt
        final_score = self.evaluate(current_prompt, task, context)
        logger.info(f"Final effectiveness score: {final_score:.4f}")
        
        # Update internal state
        self._state["optimization_history"].append({
            "base_prompt": base_prompt,
            "task": task,
            "context": context,
            "constraints": constraints,
            "final_prompt": current_prompt,
            "improvement": final_score - effectiveness_scores[0],
            "iterations": len(prompt_versions) - 1,
        })
        
        self._state["prompt_versions"] = prompt_versions
        self._state["effectiveness_scores"] = {
            "initial": effectiveness_scores[0],
            "final": final_score,
            "history": effectiveness_scores,
        }
        self._state["current_prompt"] = current_prompt
        
        if trace_optimization:
            self._state["optimization_trace"] = trace
        
        return current_prompt
    
    def _determine_optimization_steps(
        self, task: str, context: Dict[str, Any], constraints: Dict[str, Any]
    ) -> List[str]:
        """Determine which optimization steps to apply based on task and context."""
        # Default optimization steps
        default_steps = [
            "attention_optimization",
            "context_management",
            "recursive_structure",
            "drift_mitigation",
        ]
        
        # Task-specific optimization
        task_specific_steps = {
            "reasoning": [
                "attention_optimization",
                "recursive_structure",
                "context_management",
                "drift_mitigation",
            ],
            "creative": [
                "attention_optimization",
                "emergence_enhancement",
                "recursive_structure",
                "context_management",
            ],
            "educational": [
                "attention_optimization",
                "cognitive_load_management",
                "context_management",
                "recursive_structure",
            ],
            "factual": [
                "attention_optimization",
                "drift_mitigation",
                "context_management",
                "attribution_enhancement",
            ],
        }
        
        # Constraint-specific adjustments
        if constraints.get("max_tokens"):
            # Prioritize efficient information packaging
            return [
                "attention_optimization",
                "context_management",
                "information_compression",
                "drift_mitigation",
            ]
        
        elif constraints.get("audience") == "novice":
            # Prioritize cognitive load management for novices
            return [
                "cognitive_load_management",
                "attention_optimization",
                "recursive_structure",
                "context_management",
            ]
        
        # Use task-specific steps if available, otherwise default
        for task_keyword, steps in task_specific_steps.items():
            if task_keyword in task.lower():
                return steps
        
        return default_steps
    
    def _apply_optimization_steps(
        self,
        prompt: str,
        optimization_steps: List[str],
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Apply a sequence of optimization steps to the prompt."""
        optimized_prompt = prompt
        
        for step in optimization_steps:
            optimization_method = getattr(self, f"_optimize_{step}", None)
            
            if optimization_method and callable(optimization_method):
                try:
                    logger.debug(f"Applying optimization step: {step}")
                    optimized_prompt = optimization_method(
                        optimized_prompt, task, context, constraints
                    )
                except Exception as e:
                    logger.warning(f"Error applying optimization step {step}: {e}")
            else:
                logger.warning(f"Unknown optimization step: {step}")
        
        return optimized_prompt
    
    def _optimize_attention_optimization(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt for attention allocation."""
        # Break prompt into context elements
        elements = self._parse_prompt_elements(prompt)
        
        # Identify key elements that should receive focus
        key_elements = self._identify_key_elements(elements, task, context)
        
        # Use attention model to optimize focus on key elements
        optimized_elements = self.attention_model.optimize_for_focus(
            context=elements,
            target_elements=key_elements,
            constraints=constraints,
        )
        
        # Reassemble prompt from optimized elements
        optimized_prompt = self._assemble_prompt(optimized_elements)
        
        return optimized_prompt
    
    def _optimize_context_management(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt for context window management."""
        # Break prompt into context elements
        elements = self._parse_prompt_elements(prompt)
        
        # Calculate current prompt length
        current_length = len(elements)
        
        # Determine target length based on constraints
        max_elements = constraints.get("max_elements", 10)
        target_length = min(current_length, max_elements)
        
        if current_length <= target_length:
            # No need to compress
            return prompt
        
        # Calculate importance scores for elements
        importance_scores = self._calculate_element_importance(elements, task, context)
        
        # Keep only the most important elements
        important_indices = sorted(
            range(len(importance_scores)),
            key=lambda i: importance_scores[i],
            reverse=True,
        )[:target_length]
        
        important_elements = [elements[i] for i in sorted(important_indices)]
        
        # Reassemble prompt from important elements
        optimized_prompt = self._assemble_prompt(important_elements)
        
        return optimized_prompt
    
    def _optimize_recursive_structure(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt for recursive processing structure."""
        # Analyze current recursive structure
        current_structure = self._analyze_recursive_structure(prompt)
        
        # Determine target recursive structure based on task
        target_structure = self._determine_target_structure(task, context, constraints)
        
        # Apply recursive structure optimization
        optimized_prompt = self._apply_recursive_structure(
            prompt, current_structure, target_structure
        )
        
        return optimized_prompt
    
    def _optimize_drift_mitigation(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt to mitigate drift and improve stability."""
        # Check if we have a reference prompt to measure drift against
        reference_prompt = self._state.get("reference_prompt")
        if not reference_prompt:
            return prompt
        
        # Create state history from prompt versions
        state_history = [
            {"content": version} for version in self._state.get("prompt_versions", [prompt])
        ]
        
        # Measure drift
        drift_result = self.drift_model.measure_drift(
            state_history=state_history,
            reference_state={"content": reference_prompt},
        )
        
        # If drift is not significant, return prompt unchanged
        if not drift_result["significant_drift"]:
            return prompt
        
        # Apply drift mitigation based on drift type
        drift_type = drift_result["drift_type"]
        
        if drift_type.value in ["divergent", "maladaptive"]:
            # Apply stronger anchoring to reference prompt
            stabilized_prompt = self._anchor_to_reference(
                prompt, reference_prompt, strength=0.7
            )
            return stabilized_prompt
        
        elif drift_type.value == "oscillatory":
            # Apply damping to reduce oscillation
            damped_prompt = self._apply_oscillation_damping(
                prompt, state_history
            )
            return damped_prompt
        
        elif drift_type.value == "unintentional":
            # Apply mild correction toward reference
            corrected_prompt = self._anchor_to_reference(
                prompt, reference_prompt, strength=0.3
            )
            return corrected_prompt
        
        # For other drift types, return prompt unchanged
        return prompt
    
    def _optimize_cognitive_load_management(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt to manage cognitive load."""
        # Break prompt into context elements
        elements = self._parse_prompt_elements(prompt)
        
        # Determine target audience cognitive capacity
        audience = constraints.get("audience", context.get("audience", "general"))
        
        cognitive_capacities = {
            "expert": 9,
            "general": 7,
            "novice": 5,
            "child": 3,
        }
        
        target_capacity = cognitive_capacities.get(audience, 7)
        
        # Analyze current cognitive load
        current_load = self._estimate_cognitive_load(elements)
        
        # If load is already within capacity, return unchanged
        if current_load <= target_capacity:
            return prompt
        
        # Apply cognitive load reduction techniques
        load_reduced_elements = self._reduce_cognitive_load(
            elements, current_load, target_capacity
        )
        
        # Reassemble prompt from load-reduced elements
        optimized_prompt = self._assemble_prompt(load_reduced_elements)
        
        return optimized_prompt
    
    def _optimize_emergence_enhancement(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt to enhance emergent properties."""
        # Analyze emergence potential
        emergence_score = self._analyze_emergence_potential(prompt, task)
        
        # If emergence potential is already high, return unchanged
        if emergence_score >= 0.8:
            return prompt
        
        # Apply emergence enhancement techniques
        enhanced_prompt = self._apply_emergence_enhancement(
            prompt, task, context, constraints
        )
        
        return enhanced_prompt
    
    def _optimize_attribution_enhancement(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt to enhance attribution clarity."""
        # Check if attribution is relevant for this task
        if not any(keyword in task.lower() for keyword in ["factual", "research", "evidence"]):
            return prompt
        
        # Enhance attribution clarity
        enhanced_prompt = self._apply_attribution_enhancement(
            prompt, task, context, constraints
        )
        
        return enhanced_prompt
    
    def _optimize_information_compression(
        self,
        prompt: str,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> str:
        """Optimize prompt through information compression."""
        # Get token limit constraint
        max_tokens = constraints.get("max_tokens")
        if not max_tokens:
            return prompt
        
        # Estimate current token count
        current_tokens = self._estimate_token_count(prompt)
        
        # If already within limit, return unchanged
        if current_tokens <= max_tokens:
            return prompt
        
        # Apply information compression
        compressed_prompt = self._apply_information_compression(
            prompt, current_tokens, max_tokens
        )
        
        return compressed_prompt
    
    def evaluate(
        self,
        prompt: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate prompt effectiveness.
        
        This method implements the prompt effectiveness function E(p, c, g) defined 
        in Section 4.3 of the Prompt Theory mathematical framework.
        
        Args:
            prompt: Prompt to evaluate
            task: Task description
            context: Context information
            
        Returns:
            Effectiveness score (0-1)
            
        Raises:
            ValueError: If prompt is empty or invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        # Initialize context
        context = context or {}
        
        # Calculate component scores
        contextual_compatibility = self._evaluate_contextual_compatibility(prompt, context)
        task_alignment = self._evaluate_task_alignment(prompt, task)
        prompt_validity = self._evaluate_prompt_validity(prompt)
        
        # Component weights
        alpha = 0.3  # Context compatibility weight
        beta = 0.5   # Task alignment weight
        gamma = 0.2  # Prompt validity weight
        
        # Calculate overall effectiveness using the prompt effectiveness function
        effectiveness = (
            alpha * contextual_compatibility +
            beta * task_alignment +
            gamma * prompt_validity
        )
        
        return effectiveness
    
    def _evaluate_contextual_compatibility(
        self, prompt: str, context: Dict[str, Any]
    ) -> float:
        """Evaluate prompt-context compatibility."""
        # If no context provided, assume maximum compatibility
        if not context:
            return 1.0
        
        # Extract context elements
        context_keys = set(context.keys())
        
        # Count context element references in prompt
        context_references = 0
        for key in context_keys:
            if key in prompt or str(context[key]) in prompt:
                context_references += 1
        
        # Calculate compatibility score
        if not context_keys:
            compatibility = 1.0
        else:
            compatibility = min(1.0, context_references / len(context_keys))
        
        return compatibility
    
    def _evaluate_task_alignment(self, prompt: str, task: str) -> float:
        """Evaluate prompt-task alignment."""
        # Define task-specific keywords
        task_keywords = {
            "reasoning": ["explain", "reason", "analyze", "evaluate", "consider"],
            "creative": ["create", "generate", "imagine", "design", "invent"],
            "educational": ["teach", "explain", "demonstrate", "clarify", "learn"],
            "factual": ["provide", "list", "summarize", "identify", "describe"],
        }
        
        # Identify relevant keywords for this task
        relevant_keywords = []
        for task_type, keywords in task_keywords.items():
            if task_type in task.lower():
                relevant_keywords.extend(keywords)
        
        # If no specific task type identified, use general evaluation
        if not relevant_keywords:
            # Check if task is mentioned in prompt
            task_words = task.lower().split()
            task_word_count = sum(1 for word in task_words if word in prompt.lower())
            task_alignment = min(1.0, task_word_count / max(1, len(task_words)))
            return task_alignment
        
        # Count task keyword occurrences in prompt
        keyword_count = sum(1 for keyword in relevant_keywords if keyword in prompt.lower())
        
        # Calculate task alignment score
        task_alignment = min(1.0, keyword_count / max(1, len(relevant_keywords)))
        
        return task_alignment
    
    def _evaluate_prompt_validity(self, prompt: str) -> float:
        """Evaluate prompt structural validity."""
        # Check for basic structural elements
        has_clear_instruction = any(word in prompt.lower() for word in ["please", "explain", "describe", "tell", "provide", "create", "generate", "analyze"])
        has_question_mark = "?" in prompt
        has_proper_length = len(prompt) >= 10  # Minimum reasonable length
        
        # Check for coherence (simplified)
        has_coherent_structure = len(prompt.split(".")) > 1 or len(prompt.split("\n")) > 1
        
        # Calculate validity score
        validity_checks = [
            has_clear_instruction,
            has_question_mark or has_clear_instruction,  # Either is acceptable
            has_proper_length,
            has_coherent_structure,
        ]
        
        validity_score = sum(1 for check in validity_checks if check) / len(validity_checks)
        
        return validity_score
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response using the optimized prompt.
        
        Args:
            prompt: Optimized prompt
            temperature: Generation temperature. Default 0.7.
            max_tokens: Maximum tokens to generate. Default None.
            stop_sequences: Sequences that stop generation. Default None.
            additional_params: Additional model-specific parameters. Default None.
            
        Returns:
            Generated response
            
        Raises:
            ValueError: If prompt is empty or invalid
            RuntimeError: If model generation fails
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        # Initialize additional parameters
        params = additional_params or {}
        
        # Add standard parameters
        params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stop_sequences is not None:
            params["stop"] = stop_sequences
        
        try:
            # Generate response based on model type
            model_name = getattr(self.model, "model_name", str(self.model))
            
            if "openai" in str(type(self.model)).lower():
                # OpenAI API
                response = self.model.chat.completions.create(
                    model=model_name if isinstance(model_name, str) else "gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                return response.choices[0].message.content
            
            elif "anthropic" in str(type(self.model)).lower():
                # Anthropic API
                response = self.model.messages.create(
                    model=model_name if isinstance(model_name, str) else "claude-3-opus-20240229",
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                return response.content[0].text
            
            else:
                # Generic model interface - assume it has a generate method
                if hasattr(self.model, "generate"):
                    response = self.model.generate(prompt, **params)
                    return response
                else:
                    raise ValueError(
                        f"Unsupported model type: {type(self.model)}. "
                        "Model must have a generate method or be OpenAI/Anthropic compatible."
                    )
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def _parse_prompt_elements(self, prompt: str) -> List[str]:
        """Parse prompt into structural elements."""
        # First try to split by markdown headings
        if "##" in prompt:
            elements = []
            current_element = ""
            
            for line in prompt.split("\n"):
                if line.startswith("#"):
                    if current_element:
                        elements.append(current_element.strip())
                    current_element = line
                else:
                    current_element += "\n" + line
            
            if current_element:
                elements.append(current_element.strip())
            
            return elements
        
        # Then try to split by blank lines
        elif "\n\n" in prompt:
            return [element.strip() for element in prompt.split("\n\n") if element.strip()]
        
        # Then try to split by sentences
        elif "." in prompt:
            import re
            elements = re.split(r'(?<=[.!?])\s+', prompt)
            return [element.strip() for element in elements if element.strip()]
        
        # Fallback to returning the whole prompt as one element
        return [prompt]
    
    def _assemble_prompt(self, elements: List[str]) -> str:
        """Reassemble prompt from structural elements."""
        if not elements:
            return ""
        
        # Check if elements look like markdown sections
        if any(element.startswith("#") for element in elements):
            return "\n\n".join(elements)
        
        # Check if elements look like paragraphs
        if any("\n" in element for element in elements):
            return "\n\n".join(elements)
        
        # Default to space-separated sentences
        return " ".join(elements)
    
    def _identify_key_elements(
        self, elements: List[str], task: str, context: Dict[str, Any]
    ) -> List[int]:
        """Identify key elements that should receive focus."""
        key_indices = []
        
        for i, element in enumerate(elements):
            # Check for task-specific keywords
            if any(keyword in element.lower() for keyword in task.lower().split()):
                key_indices.append(i)
            
            # Check for context references
            for key, value in context.items():
                if key in element or str(value) in element:
                    key_indices.append(i)
                    break
            
            # Check for structural importance
            if element.startswith("#") or i == 0:  # Heading or first element
                key_indices.append(i)
        
        # Remove duplicates
        return list(set(key_indices))
    
    def _calculate_element_importance(
        self, elements: List[str], task: str, context: Dict[str, Any]
    ) -> List[float]:
        """Calculate importance scores for prompt elements."""
        importance_scores = []
        
        for i, element in enumerate(elements):
            score = 0.0
            
            # Task relevance
            task_words = task.lower().split()
            task_relevance = sum(1 for word in task_words if word in element.lower())
            score += 0.5 * min(1.0, task_relevance / max(1, len(task_words)))
            
            # Context relevance
            context_relevance = 0.0
            for key, value in context.items():
                if key in element or str(value) in element:
                    context_relevance += 0.2
            score += min(1.0, context_relevance)
            
            # Structural importance
            if element.startswith("#"):  # Heading
                heading_level = len(element) - len(element.lstrip("#"))
                score += 1
