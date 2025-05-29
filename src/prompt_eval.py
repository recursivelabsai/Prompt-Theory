"""
Prompt effectiveness evaluation module for the Prompt Theory framework.

This module implements the prompt effectiveness function E(p, c, g) defined in Section 4.3
of the Prompt Theory mathematical framework, providing a unified approach to evaluating
prompt quality across both AI and human cognitive systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import re
import warnings

from prompt_theory.models.attention import AttentionModel
from prompt_theory.models.recursion import RecursiveProcessor
from prompt_theory.models.drift import DriftModel
from prompt_theory.utils.logging import get_logger
from prompt_theory.utils.validation import validate_parameters


logger = get_logger(__name__)


class PromptEffectivenessEvaluator:
    """Evaluates prompt effectiveness across dimensions.
    
    This class implements the prompt effectiveness function E(p, c, g) defined in
    Section 4.3 of the Prompt Theory mathematical framework, which quantifies how
    well a prompt will perform for a given task and context.
    
    Attributes:
        model: Optional LLM model for model-based evaluations
        attention_model: Model for attention allocation analysis
        recursive_processor: Model for recursive processing analysis
        drift_model: Model for drift and stability analysis
        _state: Internal state tracking evaluation history
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        attention_model: Optional[AttentionModel] = None,
        recursive_processor: Optional[RecursiveProcessor] = None,
        drift_model: Optional[DriftModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize effectiveness evaluator.
        
        Args:
            model: LLM model for evaluations (optional)
            attention_model: Attention allocation model
            recursive_processor: Recursive processing model
            drift_model: Drift and stability model
            config: Additional configuration parameters
        """
        self.config = config or {}
        self.model = model
        
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
            "evaluation_history": [],
            "component_scores": {},
            "effectiveness_models": {},
        }
        
        # Initialize effectiveness component weights
        self._component_weights = {
            "contextual_compatibility": self.config.get("contextual_weight", 0.3),
            "task_alignment": self.config.get("task_weight", 0.5),
            "prompt_validity": self.config.get("validity_weight", 0.2),
        }
        
        logger.debug(f"Initialized PromptEffectivenessEvaluator with weights: {self._component_weights}")
    
    def evaluate(
        self,
        prompt: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        detailed_scores: bool = False,
    ) -> Union[float, Dict[str, Any]]:
        """Evaluate overall prompt effectiveness.
        
        This method implements the prompt effectiveness function E(p, c, g) defined in
        Section 4.3 of the Prompt Theory mathematical framework:
        
        E(p, c, g) = α · C(p, c) + β · T(p, g) + γ · V(p)
        
        Args:
            prompt: Prompt to evaluate
            task: Task description
            context: Context information
            detailed_scores: Whether to return detailed component scores
            
        Returns:
            Effectiveness score (0-1) or dictionary with detailed scores
            
        Raises:
            ValueError: If prompt is empty or invalid
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        # Initialize context
        context = context or {}
        
        # Calculate component scores
        compatibility_score = self.evaluate_contextual_compatibility(prompt, context)
        alignment_score = self.evaluate_task_alignment(prompt, task)
        validity_score = self.evaluate_prompt_validity(prompt)
        
        # Calculate composite effectiveness score using weighted sum
        effectiveness = (
            self._component_weights["contextual_compatibility"] * compatibility_score +
            self._component_weights["task_alignment"] * alignment_score +
            self._component_weights["prompt_validity"] * validity_score
        )
        
        # Update state
        self._state["evaluation_history"].append({
            "prompt": prompt,
            "task": task,
            "context": context,
            "scores": {
                "contextual_compatibility": compatibility_score,
                "task_alignment": alignment_score,
                "prompt_validity": validity_score,
                "overall_effectiveness": effectiveness,
            },
        })
        
        # Return detailed scores if requested
        if detailed_scores:
            return {
                "overall_effectiveness": effectiveness,
                "component_scores": {
                    "contextual_compatibility": compatibility_score,
                    "task_alignment": alignment_score,
                    "prompt_validity": validity_score,
                },
                "weights": self._component_weights.copy(),
                "analysis": self._analyze_effectiveness(
                    prompt, compatibility_score, alignment_score, validity_score
                ),
            }
        
        return effectiveness
    
    def evaluate_contextual_compatibility(
        self, prompt: str, context: Dict[str, Any]
    ) -> float:
        """Evaluate prompt-context compatibility.
        
        This method implements the context compatibility function C(p, c) from
        the prompt effectiveness function.
        
        Args:
            prompt: Prompt to evaluate
            context: Context information
            
        Returns:
            Compatibility score (0-1)
        """
        # If no context provided, assume neutral compatibility
        if not context:
            return 0.5
        
        # Extract context elements
        context_keys = set(context.keys())
        context_values = []
        for value in context.values():
            if isinstance(value, str):
                context_values.append(value)
            elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                context_values.extend(value)
        
        # Calculate key reference score
        key_references = 0
        for key in context_keys:
            if key.lower() in prompt.lower():
                key_references += 1
        
        key_reference_score = key_references / max(1, len(context_keys))
        
        # Calculate value reference score
        value_references = 0
        for value in context_values:
            if value.lower() in prompt.lower():
                value_references += 1
        
        value_reference_score = value_references / max(1, len(context_values))
        
        # Calculate semantic alignment using attention model
        semantic_alignment = self._calculate_semantic_alignment(prompt, context)
        
        # Combine scores with weights
        compatibility = (
            0.3 * key_reference_score +
            0.3 * value_reference_score +
            0.4 * semantic_alignment
        )
        
        return compatibility
    
    def evaluate_task_alignment(self, prompt: str, task: str) -> float:
        """Evaluate prompt-task alignment.
        
        This method implements the task alignment function T(p, g) from
        the prompt effectiveness function.
        
        Args:
            prompt: Prompt to evaluate
            task: Task description
            
        Returns:
            Alignment score (0-1)
        """
        # Task-specific keywords
        task_type_keywords = {
            "reasoning": ["explain", "reason", "analyze", "evaluate", "consider", "examine", "investigate"],
            "creative": ["create", "generate", "imagine", "design", "invent", "develop", "compose"],
            "educational": ["teach", "explain", "demonstrate", "clarify", "learn", "understand", "instruct"],
            "factual": ["provide", "list", "summarize", "identify", "describe", "outline", "enumerate"],
            "decision": ["decide", "choose", "select", "determine", "assess", "judge", "compare"],
            "planning": ["plan", "organize", "arrange", "schedule", "prepare", "strategize", "coordinate"],
            "problem_solving": ["solve", "resolve", "address", "overcome", "fix", "tackle", "handle"],
        }
        
        # Determine task type
        task_type = None
        max_keyword_matches = 0
        
        for type_name, keywords in task_type_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in task.lower())
            if matches > max_keyword_matches:
                max_keyword_matches = matches
                task_type = type_name
        
        # If no clear task type identified, use general approach
        if task_type is None:
            # Calculate direct word overlap between task and prompt
            task_words = set(re.findall(r'\b\w+\b', task.lower()))
            prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
            
            overlap = len(task_words.intersection(prompt_words))
            task_coverage = overlap / max(1, len(task_words))
            
            # Simple alignment score based on word overlap
            return min(1.0, task_coverage * 2)  # Scale up for reasonable baseline
        
        # Task type specific evaluation
        relevant_keywords = task_type_keywords[task_type]
        
        # Check for task-specific keywords in prompt
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword.lower() in prompt.lower())
        keyword_score = min(1.0, keyword_matches / max(1, len(relevant_keywords) * 0.5))
        
        # Check for task description words in prompt
        task_words = set(re.findall(r'\b\w+\b', task.lower()))
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", "and", "or"}
        task_words = task_words.difference(stop_words)
        
        # Calculate task description coverage
        overlap = len(task_words.intersection(prompt_words))
        description_coverage = overlap / max(1, len(task_words))
        
        # Check for task-specific structures
        structure_score = self._evaluate_task_specific_structure(prompt, task_type)
        
        # Combine scores with weights
        alignment = (
            0.4 * keyword_score +
            0.4 * description_coverage +
            0.2 * structure_score
        )
        
        return alignment
    
    def evaluate_prompt_validity(self, prompt: str) -> float:
        """Evaluate prompt structural validity.
        
        This method implements the prompt validity function V(p) from
        the prompt effectiveness function.
        
        Args:
            prompt: Prompt to evaluate
            
        Returns:
            Validity score (0-1)
        """
        # Initialize validity metrics
        metrics = {}
        
        # Check clarity and structure
        metrics["has_clear_instruction"] = any(word in prompt.lower() for word in 
            ["please", "explain", "describe", "tell", "provide", "create", "generate", "analyze", "list"])
        
        metrics["has_question"] = "?" in prompt
        metrics["has_proper_length"] = len(prompt) >= 10
        
        # Check for coherent structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', prompt) if s.strip()]
        paragraphs = [p.strip() for p in prompt.split("\n\n") if p.strip()]
        
        metrics["has_multiple_sentences"] = len(sentences) > 1
        metrics["has_reasonable_sentence_length"] = all(3 <= len(s.split()) <= 50 for s in sentences if s)
        metrics["has_structured_paragraphs"] = len(paragraphs) > 1 if len(prompt) > 200 else True
        
        # Check for formatting elements
        metrics["has_formatting"] = any(marker in prompt for marker in ["#", "*", "-", "1.", "2.", "•"])
        
        # Check for potential issues
        metrics["no_excessive_punctuation"] = not re.search(r'([!?.]{2,})', prompt)
        metrics["no_excessive_repetition"] = not re.search(r'\b(\w+)\s+\1\s+\1\b', prompt.lower())
        
        # Calculate overall validity score
        validity_score = sum(1 for metric_value in metrics.values() if metric_value) / len(metrics)
        
        # Store metrics for analysis
        self._state["component_scores"]["validity_metrics"] = metrics
        
        return validity_score
    
    def _calculate_semantic_alignment(
        self, prompt: str, context: Dict[str, Any]
    ) -> float:
        """Calculate semantic alignment between prompt and context."""
        # Without embeddings, use a simplified approach based on key concept overlap
        
        # Extract key concepts from context
        context_concepts = set()
        for key, value in context.items():
            # Add key as concept
            context_concepts.add(key.lower())
            
            # Add values as concepts if they're strings
            if isinstance(value, str):
                # Extract substantive words
                words = re.findall(r'\b[a-zA-Z]{4,}\b', value.lower())
                context_concepts.update(words)
            elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                for v in value:
                    words = re.findall(r'\b[a-zA-Z]{4,}\b', v.lower())
                    context_concepts.update(words)
        
        # Extract key concepts from prompt
        prompt_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', prompt.lower()))
        
        # Calculate overlap
        if not context_concepts:
            return 0.5  # Neutral score if no concepts extracted
        
        overlap = len(prompt_words.intersection(context_concepts))
        alignment = min(1.0, overlap / max(5, len(context_concepts) * 0.3))  # Expect ~30% concept coverage
        
        return alignment
    
    def _evaluate_task_specific_structure(self, prompt: str, task_type: str) -> float:
        """Evaluate whether prompt has appropriate structure for the task type."""
        # Define task-specific structural expectations
        structure_patterns = {
            "reasoning": {
                "patterns": [
                    r'(?:first|1st|initially|begin by|start)',  # Sequential reasoning indicators
                    r'(?:second|2nd|next|then|after that)',
                    r'(?:third|3rd|finally|lastly|conclude)',
                    r'(?:because|since|as|therefore|thus|hence|consequently)',  # Causal indicators
                    r'(?:if|when|suppose|assuming|given that)',  # Conditional indicators
                ],
                "section_headers": ["premise", "argument", "analysis", "conclusion"],
            },
            "creative": {
                "patterns": [
                    r'(?:creative|innovative|novel|original|imaginative)',  # Creativity indicators
                    r'(?:develop|explore|experiment|play with|try)',  # Exploration indicators
                    r'(?:inspire|inspiration|inspired by)',  # Inspiration indicators
                ],
                "section_headers": ["concept", "idea", "theme", "style", "approach"],
            },
            "educational": {
                "patterns": [
                    r'(?:explain|clarify|define|describe|elaborate)',  # Explanation indicators
                    r'(?:example|instance|illustration|demonstrate)',  # Example indicators
                    r'(?:concept|principle|theory|framework|model)',  # Conceptual indicators
                    r'(?:understand|comprehend|grasp|learn)',  # Learning indicators
                ],
                "section_headers": ["introduction", "concept", "explanation", "example", "summary"],
            },
            "factual": {
                "patterns": [
                    r'(?:list|enumerate|outline|summarize)',  # List indicators
                    r'(?:fact|information|data|statistic|figure)',  # Factual indicators
                    r'(?:accurate|precise|exact|specific|detailed)',  # Precision indicators
                ],
                "section_headers": ["background", "facts", "information", "details", "summary"],
            },
            "decision": {
                "patterns": [
                    r'(?:option|alternative|choice|possibility)',  # Option indicators
                    r'(?:criteria|factor|consideration|aspect)',  # Criteria indicators
                    r'(?:compare|contrast|weigh|evaluate|assess)',  # Comparison indicators
                    r'(?:decide|select|choose|determine|pick)',  # Decision indicators
                ],
                "section_headers": ["options", "criteria", "analysis", "recommendation", "decision"],
            },
            "planning": {
                "patterns": [
                    r'(?:step|phase|stage|task|action)',  # Step indicators
                    r'(?:timeline|schedule|timeframe|deadline)',  # Timeline indicators
                    r'(?:resource|material|tool|equipment)',  # Resource indicators
                    r'(?:goal|objective|target|outcome)',  # Goal indicators
                ],
                "section_headers": ["goals", "steps", "timeline", "resources", "outcome"],
            },
            "problem_solving": {
                "patterns": [
                    r'(?:problem|issue|challenge|difficulty)',  # Problem indicators
                    r'(?:cause|source|origin|reason)',  # Cause indicators
                    r'(?:solution|resolve|fix|address|overcome)',  # Solution indicators
                    r'(?:implement|execute|carry out|apply)',  # Implementation indicators
                ],
                "section_headers": ["problem", "cause", "solution", "implementation", "validation"],
            },
        }
        
        # Get expected patterns for this task type
        if task_type not in structure_patterns:
            return 0.5  # Neutral score for unknown task types
        
        expected_patterns = structure_patterns[task_type]["patterns"]
        expected_headers = structure_patterns[task_type]["section_headers"]
        
        # Check for pattern matches
        pattern_matches = sum(1 for pattern in expected_patterns if re.search(pattern, prompt.lower()))
        pattern_score = min(1.0, pattern_matches / max(1, len(expected_patterns)))
        
        # Check for section headers
        header_pattern = r'(?:^|\n)(?:#+ |## )([a-zA-Z]+)'
        headers = re.findall(header_pattern, prompt, re.IGNORECASE)
        
        header_matches = 0
        for header in headers:
            if any(expected in header.lower() for expected in expected_headers):
                header_matches += 1
        
        header_score = min(1.0, header_matches / max(1, len(headers))) if headers else 0.0
        
        # Combined structure score
        return 0.7 * pattern_score + 0.3 * header_score
    
    def _analyze_effectiveness(
        self,
        prompt: str,
        compatibility_score: float,
        alignment_score: float,
        validity_score: float,
    ) -> Dict[str, Any]:
        """Analyze effectiveness scores to provide insights and recommendations."""
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
        }
        
        # Analyze compatibility score
        if compatibility_score >= 0.8:
            analysis["strengths"].append("Excellent context integration")
        elif compatibility_score >= 0.6:
            analysis["strengths"].append("Good context integration")
        elif compatibility_score < 0.4:
            analysis["weaknesses"].append("Poor context integration")
            analysis["recommendations"].append("Incorporate more context-specific information and terminology")
        
        # Analyze alignment score
        if alignment_score >= 0.8:
            analysis["strengths"].append("Excellent task alignment")
        elif alignment_score >= 0.6:
            analysis["strengths"].append("Good task alignment")
        elif alignment_score < 0.4:
            analysis["weaknesses"].append("Poor task alignment")
            analysis["recommendations"].append("Clarify the task objective and include relevant task-specific terms")
        
        # Analyze validity score
        if validity_score >= 0.8:
            analysis["strengths"].append("Well-structured prompt")
        elif validity_score >= 0.6:
            analysis["strengths"].append("Adequately structured prompt")
        elif validity_score < 0.4:
            analysis["weaknesses"].append("Poorly structured prompt")
            analysis["recommendations"].append("Improve prompt clarity, organization, and formatting")
        
        # Overall effectiveness analysis
        overall_score = (
            self._component_weights["contextual_compatibility"] * compatibility_score +
            self._component_weights["task_alignment"] * alignment_score +
            self._component_weights["prompt_validity"] * validity_score
        )
        
        if overall_score >= 0.8:
            analysis["summary"] = "Highly effective prompt"
        elif overall_score >= 0.6:
            analysis["summary"] = "Effective prompt with room for improvement"
        elif overall_score >= 0.4:
            analysis["summary"] = "Moderately effective prompt requiring significant improvements"
        else:
            analysis["summary"] = "Ineffective prompt requiring major revision"
        
        return analysis
    
    def compare_prompts(
        self,
        prompts: List[str],
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple prompts for the same task and context.
        
        Args:
            prompts: List of prompts to compare
            task: Task description
            context: Context information
            
        Returns:
            Dictionary with comparison results
            
        Raises:
            ValueError: If prompts list is empty
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        # Initialize context
        context = context or {}
        
        # Evaluate each prompt
        evaluations = []
        for i, prompt in enumerate(prompts):
            evaluation = self.evaluate(prompt, task, context, detailed_scores=True)
            evaluations.append({
                "prompt_index": i,
                "prompt": prompt,
                "evaluation": evaluation,
            })
        
        # Sort by overall effectiveness
        sorted_evaluations = sorted(
            evaluations,
            key=lambda x: x["evaluation"]["overall_effectiveness"],
            reverse=True,
        )
        
        # Identify best prompt
        best_prompt = sorted_evaluations[0]
        
        # Calculate comparative statistics
        avg_effectiveness = sum(e["evaluation"]["overall_effectiveness"] for e in evaluations) / len(evaluations)
        std_effectiveness = np.std([e["evaluation"]["overall_effectiveness"] for e in evaluations])
        
        # Generate comparison report
        comparison = {
            "best_prompt_index": best_prompt["prompt_index"],
            "best_prompt": best_prompt["prompt"],
            "best_prompt_score": best_prompt["evaluation"]["overall_effectiveness"],
            "average_effectiveness": avg_effectiveness,
            "std_effectiveness": std_effectiveness,
            "prompt_rankings": [
                {
                    "rank": i + 1,
                    "prompt_index": e["prompt_index"],
                    "effectiveness": e["evaluation"]["overall_effectiveness"],
                    "strengths": e["evaluation"]["analysis"]["strengths"],
                    "weaknesses": e["evaluation"]["analysis"]["weaknesses"],
                }
                for i, e in enumerate(sorted_evaluations)
            ],
            "component_analysis": {
                "contextual_compatibility": {
                    "best_prompt": max(evaluations, key=lambda x: x["evaluation"]["component_scores"]["contextual_compatibility"])["prompt_index"],
                    "worst_prompt": min(evaluations, key=lambda x: x["evaluation"]["component_scores"]["contextual_compatibility"])["prompt_index"],
                },
                "task_alignment": {
                    "best_prompt": max(evaluations, key=lambda x: x["evaluation"]["component_scores"]["task_alignment"])["prompt_index"],
                    "worst_prompt": min(evaluations, key=lambda x: x["evaluation"]["component_scores"]["task_alignment"])["prompt_index"],
                },
                "prompt_validity": {
                    "best_prompt": max(evaluations, key=lambda x: x["evaluation"]["component_scores"]["prompt_validity"])["prompt_index"],
                    "worst_prompt": min(evaluations, key=lambda x: x["evaluation"]["component_scores"]["prompt_validity"])["prompt_index"],
                },
            },
        }
        
        return comparison
    
    def evaluate_human_comprehension(
        self,
        prompt: str,
        comprehension_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate expected human comprehension of a prompt.
        
        Args:
            prompt: Prompt to evaluate
            comprehension_metrics: Optional human comprehension metrics (if available)
            
        Returns:
            Dictionary with human comprehension evaluation
        """
        # Calculate readability metrics
        readability = self._calculate_readability_metrics(prompt)
        
        # Estimate cognitive load
        cognitive_load = self._estimate_cognitive_load(prompt)
        
        # Analyze structure for human comprehension
        structure_analysis = self._analyze_structure_for_comprehension(prompt)
        
        # Combine metrics into overall comprehension score
        comprehension_score = (
            0.4 * (1 - readability["normalized_flesch_kincaid"]) +  # Lower grade level is better
            0.3 * (1 - cognitive_load["overall_load"]) +  # Lower cognitive load is better
            0.3 * structure_analysis["comprehension_support"]  # Higher structure support is better
        )
        
        # Generate human comprehension evaluation
        evaluation = {
            "overall_comprehension_score": comprehension_score,
            "readability": readability,
            "cognitive_load": cognitive_load,
            "structure_analysis": structure_analysis,
            "audience_suitability": self._evaluate_audience_suitability(
                readability["flesch_kincaid_grade"], cognitive_load["overall_load"]
            ),
        }
        
        # Incorporate human metrics if provided
        if comprehension_metrics:
            evaluation["human_metrics"] = comprehension_metrics
            
            # Adjust overall score based on human metrics
            if "comprehension_score" in comprehension_metrics:
                evaluation["overall_comprehension_score"] = (
                    0.5 * evaluation["overall_comprehension_score"] +
                    0.5 * comprehension_metrics["comprehension_score"]
                )
        
        return evaluation
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for text."""
        # Count sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        
        # Count words
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Count syllables (simplified approach)
        def count_syllables(word):
            word = word.lower()
            # Exception for words ending in 'e' or 'es'
            if word.endswith('e'):
                word = word[:-1]
            elif word.endswith('es'):
                word = word[:-2]
            
            # Count vowel groups
            vowels = "aeiouy"
            count = 0
            in_vowel_group = False
            
            for char in word:
                if char in vowels:
                    if not in_vowel_group:
                        count += 1
                        in_vowel_group = True
                else:
                    in_vowel_group = False
            
            # Ensure at least one syllable
            return max(1, count)
        
        syllable_count = sum(count_syllables(word) for word in words)
        
        # Calculate Flesch-Kincaid Grade Level
        if sentence_count == 0 or word_count == 0:
            flesch_kincaid_grade = 12  # Default to high grade level for invalid input
        else:
            flesch_kincaid_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
            flesch_kincaid_grade = max(0, min(18, flesch_kincaid_grade))  # Clamp to reasonable range
        
        # Normalize to 0-1 scale (where 1 is most complex)
        normalized_flesch_kincaid = flesch_kincaid_grade / 18.0
        
        return {
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "normalized_flesch_kincaid": normalized_flesch_kincaid,
            "sentence_count": sentence_count,
            "word_count": word_count,
            "syllable_count": syllable_count,
            "words_per_sentence": word_count / max(1, sentence_count),
            "syllables_per_word": syllable_count / max(1, word_count),
        }
    
    def _estimate_cognitive_load(self, text: str) -> Dict[str, float]:
        """Estimate cognitive load of text based on structure and complexity."""
        # Count elements that increase cognitive load
        element_count = len(self._parse_prompt_elements(text))
        
        # Calculate information density
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        
        # Higher ratio of unique words increases load
        lexical_density = unique_words / max(1, word_count)
        
        # Count complex terms (simplified approach)
        complex_terms = [
            "therefore", "consequently", "subsequently", "nevertheless",
            "furthermore", "additionally", "notwithstanding", "accordingly",
            "alternatively", "simultaneously", "fundamentally", "theoretically",
        ]
        
        complex_term_count = sum(term in text.lower() for term in complex_terms)
        complex_term_ratio = complex_term_count / max(1, word_count / 100)  # Normalize by 100 words
        
        # Count nested structures
        list_depth = text.count("\n  -") + text.count("\n    -")  # Count indented list items
        section_depth = len(re.findall(r'###', text))  # Count level 3 headers
        
        # Calculate structural complexity
        structural_complexity = min(1.0, (list_depth + section_depth) / 10
