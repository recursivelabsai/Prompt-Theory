# Prompt Theory: Python Package Structure

This document outlines the comprehensive structure of the `prompt-theory` Python package, detailing the purpose and contents of each module, class, and function.

## Core Package Structure

```
prompt_theory/
├── __init__.py                # Package initialization and version info
├── models/                    # Core theoretical models
├── optimizers/                # Prompt optimization tools
├── evaluation/                # Evaluation metrics and tools
├── visualization/             # Visualization utilities
├── applications/              # Domain-specific applications
└── utils/                     # Utility functions and helpers
```

## Detailed Module Structure

### 1. Core Models (`prompt_theory/models/`)

```
models/
├── __init__.py
├── attention.py               # Attention allocation models
├── recursion.py               # Recursive processing models
├── drift.py                   # Drift and stability models
├── emergence.py               # Emergence and collapse models
├── information_processing.py  # Information processing primitives
└── base.py                    # Abstract base classes
```

#### Key Classes in `models/`

##### `attention.py`

```python
class AttentionModel:
    """Models attention allocation in both AI and human systems."""
    
    def __init__(self, capacity=None, recency_bias=0.3, salience_weight=0.5):
        """Initialize attention model parameters.
        
        Args:
            capacity (float, optional): Attention capacity limit. Default to system-specific value.
            recency_bias (float): Weight for recency bias (0-1). Default 0.3.
            salience_weight (float): Weight for salience factors (0-1). Default 0.5.
        """
        
    def allocate(self, context, queries, keys, values):
        """Allocate attention across context elements.
        
        Args:
            context (list): List of context elements
            queries (list): List of query vectors
            keys (list): List of key vectors
            values (list): List of value vectors
            
        Returns:
            dict: Attention allocation across context elements
        """
        
    def predict_human_attention(self, stimuli, task, individual_params=None):
        """Predict human attention allocation for given stimuli.
        
        Args:
            stimuli (list): List of stimuli
            task (str): Task description
            individual_params (dict, optional): Individual-specific parameters
            
        Returns:
            dict: Predicted attention allocation
        """
        
    def optimize_for_focus(self, context, target_elements):
        """Optimize context to focus attention on target elements.
        
        Args:
            context (list): Current context elements
            target_elements (list): Elements to focus attention on
            
        Returns:
            list: Optimized context
        """
```

##### `recursion.py`

```python
class RecursiveProcessor:
    """Models recursive processing in both AI and human systems."""
    
    def __init__(self, max_depth=5, collapse_threshold=0.8, emergence_threshold=0.6):
        """Initialize recursive processor.
        
        Args:
            max_depth (int): Maximum recursion depth. Default 5.
            collapse_threshold (float): Threshold for recursive collapse (0-1). Default 0.8.
            emergence_threshold (float): Threshold for emergence (0-1). Default 0.6.
        """
        
    def process(self, input_data, initial_state=None, max_iterations=10):
        """Process input recursively.
        
        Args:
            input_data: Input to process
            initial_state (dict, optional): Initial system state
            max_iterations (int): Maximum processing iterations
            
        Returns:
            dict: Final state and output
        """
        
    def detect_collapse(self, state_history):
        """Detect recursive collapse based on state history.
        
        Args:
            state_history (list): History of system states
            
        Returns:
            bool: True if collapse detected
        """
        
    def detect_emergence(self, state_history):
        """Detect emergence based on state history.
        
        Args:
            state_history (list): History of system states
            
        Returns:
            dict: Emergence detection results
        """
```

##### `drift.py`

```python
class DriftModel:
    """Models drift and stability in recursive systems."""
    
    def __init__(self, stability_params=None, drift_detection_threshold=0.3):
        """Initialize drift model.
        
        Args:
            stability_params (dict, optional): Parameters for stability calculation
            drift_detection_threshold (float): Threshold for drift detection (0-1). Default 0.3.
        """
        
    def measure_drift(self, state_history, reference_state=None):
        """Measure drift from reference state.
        
        Args:
            state_history (list): History of system states
            reference_state (dict, optional): Reference state for drift calculation
            
        Returns:
            float: Measured drift
        """
        
    def decompose_drift(self, state_history, goal_vector):
        """Decompose drift into intentional and unintentional components.
        
        Args:
            state_history (list): History of system states
            goal_vector: Vector representing goal direction
            
        Returns:
            dict: Decomposed drift measurements
        """
        
    def predict_drift(self, current_state, inputs, n_steps=5):
        """Predict future drift based on current state and inputs.
        
        Args:
            current_state (dict): Current system state
            inputs (list): Sequence of future inputs
            n_steps (int): Number of steps to predict
            
        Returns:
            list: Predicted future states
        """
```

### 2. Optimizers (`prompt_theory/optimizers/`)

```
optimizers/
├── __init__.py
├── base.py                    # Base optimizer class
├── neurobiological.py         # Bio-inspired optimizers
├── hybrid.py                  # Combined approaches
├── multi_objective.py         # Multi-objective optimization
└── constraints.py             # Constraint-based optimization
```

#### Key Classes in `optimizers/`

##### `base.py`

```python
class PromptOptimizer:
    """Base class for prompt optimization."""
    
    def __init__(self, model, attention_model=None, recursive_processor=None, drift_model=None):
        """Initialize prompt optimizer.
        
        Args:
            model (str): LLM model identifier
            attention_model (AttentionModel, optional): Attention allocation model
            recursive_processor (RecursiveProcessor, optional): Recursive processing model
            drift_model (DriftModel, optional): Drift and stability model
        """
        
    def optimize(self, base_prompt, task, context=None, constraints=None):
        """Optimize prompt for given task and context.
        
        Args:
            base_prompt (str): Base prompt to optimize
            task (str): Task description
            context (dict, optional): Context information
            constraints (dict, optional): Optimization constraints
            
        Returns:
            str: Optimized prompt
        """
        
    def evaluate(self, prompt, task, context=None):
        """Evaluate prompt effectiveness.
        
        Args:
            prompt (str): Prompt to evaluate
            task (str): Task description
            context (dict, optional): Context information
            
        Returns:
            float: Effectiveness score (0-1)
        """
        
    def generate(self, prompt):
        """Generate response using the optimized prompt.
        
        Args:
            prompt (str): Optimized prompt
            
        Returns:
            str: Generated response
        """
```

##### `neurobiological.py`

```python
class NeurobiologicalOptimizer(PromptOptimizer):
    """Optimizer inspired by neurobiological principles."""
    
    def __init__(self, model, cognitive_parameters=None, **kwargs):
        """Initialize neurobiological optimizer.
        
        Args:
            model (str): LLM model identifier
            cognitive_parameters (dict, optional): Cognitive model parameters
            **kwargs: Additional parameters for base optimizer
        """
        
    def optimize_working_memory(self, prompt, capacity=4):
        """Optimize prompt for working memory constraints.
        
        Args:
            prompt (str): Prompt to optimize
            capacity (int): Working memory capacity (chunks). Default 4.
            
        Returns:
            str: Working memory optimized prompt
        """
        
    def optimize_attention_guidance(self, prompt, key_elements):
        """Optimize prompt to guide attention to key elements.
        
        Args:
            prompt (str): Prompt to optimize
            key_elements (list): Key elements to focus attention on
            
        Returns:
            str: Attention-optimized prompt
        """
        
    def optimize_recursive_depth(self, prompt, target_depth=3):
        """Optimize prompt for specific recursive processing depth.
        
        Args:
            prompt (str): Prompt to optimize
            target_depth (int): Target recursion depth. Default 3.
            
        Returns:
            str: Recursion-optimized prompt
        """
```

### 3. Evaluation (`prompt_theory/evaluation/`)

```
evaluation/
├── __init__.py
├── effectiveness.py           # Prompt effectiveness metrics
├── collapse.py                # Collapse detection
├── emergence.py               # Emergence detection
├── human_metrics.py           # Human cognitive metrics
└── comparative.py             # AI-human comparative metrics
```

#### Key Classes in `evaluation/`

##### `effectiveness.py`

```python
class PromptEffectivenessEvaluator:
    """Evaluates prompt effectiveness across dimensions."""
    
    def __init__(self, model=None):
        """Initialize effectiveness evaluator.
        
        Args:
            model (str, optional): LLM model for evaluations
        """
        
    def evaluate(self, prompt, task, context=None):
        """Evaluate overall prompt effectiveness.
        
        Args:
            prompt (str): Prompt to evaluate
            task (str): Task description
            context (dict, optional): Context information
            
        Returns:
            dict: Effectiveness scores across dimensions
        """
        
    def evaluate_contextual_compatibility(self, prompt, context):
        """Evaluate prompt-context compatibility.
        
        Args:
            prompt (str): Prompt to evaluate
            context (dict): Context information
            
        Returns:
            float: Compatibility score (0-1)
        """
        
    def evaluate_task_alignment(self, prompt, task):
        """Evaluate prompt-task alignment.
        
        Args:
            prompt (str): Prompt to evaluate
            task (str): Task description
            
        Returns:
            float: Alignment score (0-1)
        """
        
    def evaluate_prompt_validity(self, prompt):
        """Evaluate prompt structural validity.
        
        Args:
            prompt (str): Prompt to evaluate
            
        Returns:
            float: Validity score (0-1)
        """
```

##### `collapse.py`

```python
class CollapseDetector:
    """Detects recursive collapse in system responses."""
    
    def __init__(self, collapse_threshold=0.8):
        """Initialize collapse detector.
        
        Args:
            collapse_threshold (float): Threshold for collapse detection (0-1). Default 0.8.
        """
        
    def detect(self, response, prompt=None, state_history=None):
        """Detect collapse in system response.
        
        Args:
            response (str): System response
            prompt (str, optional): Input prompt
            state_history (list, optional): History of system states
            
        Returns:
            dict: Collapse detection results
        """
        
    def analyze_collapse_type(self, response, prompt=None):
        """Analyze type of collapse if detected.
        
        Args:
            response (str): System response
            prompt (str, optional): Input prompt
            
        Returns:
            str: Collapse type identification
        """
        
    def measure_discontinuity(self, current_state, previous_state):
        """Measure state discontinuity for collapse detection.
        
        Args:
            current_state (dict): Current system state
            previous_state (dict): Previous system state
            
        Returns:
            float: Discontinuity measure (0-1)
        """
```

### 4. Visualization (`prompt_theory/visualization/`)

```
visualization/
├── __init__.py
├── attention_maps.py          # Attention visualization
├── recursive_graphs.py        # Recursion visualization
├── stability_plots.py         # Stability visualization
└── comparative.py             # Comparative visualizations
```

#### Key Classes in `visualization/`

##### `attention_maps.py`

```python
class AttentionVisualizer:
    """Visualizes attention patterns."""
    
    def __init__(self, colormap='viridis'):
        """Initialize attention visualizer.
        
        Args:
            colormap (str): Matplotlib colormap name. Default 'viridis'.
        """
        
    def plot_attention_map(self, attention_weights, labels=None):
        """Plot attention heatmap.
        
        Args:
            attention_weights (array): Attention weight matrix
            labels (list, optional): Labels for attention elements
            
        Returns:
            matplotlib.figure.Figure: Attention map figure
        """
        
    def plot_comparative_attention(self, ai_attention, human_attention, labels=None):
        """Plot comparative AI vs human attention.
        
        Args:
            ai_attention (array): AI system attention weights
            human_attention (array): Human attention weights
            labels (list, optional): Labels for attention elements
            
        Returns:
            matplotlib.figure.Figure: Comparative attention figure
        """
        
    def plot_attention_flow(self, attention_sequence, labels=None):
        """Plot attention flow over time/sequence.
        
        Args:
            attention_sequence (list): Sequence of attention weight matrices
            labels (list, optional): Labels for attention elements
            
        Returns:
            matplotlib.figure.Figure: Attention flow figure
        """
```

### 5. Applications (`prompt_theory/applications/`)

```
applications/
├── __init__.py
├── education.py               # Educational applications
├── clinical.py                # Clinical applications
├── collaborative.py           # Collaborative systems
├── interface_design.py        # Interface design applications
└── research.py                # Research applications
```

#### Key Classes in `applications/`

##### `education.py`

```python
class EducationalPromptDesigner:
    """Designs educational prompts based on Prompt Theory."""
    
    def __init__(self, optimizer=None):
        """Initialize educational prompt designer.
        
        Args:
            optimizer (PromptOptimizer, optional): Prompt optimizer
        """
        
    def design_instructional_prompt(self, topic, student_level, learning_objectives):
        """Design instructional prompt.
        
        Args:
            topic (str): Topic to teach
            student_level (str): Student level/background
            learning_objectives (list): Learning objectives
            
        Returns:
            str: Optimized instructional prompt
        """
        
    def design_scaffolded_prompt(self, concept, complexity, scaffolding_levels=3):
        """Design scaffolded learning prompt.
        
        Args:
            concept (str): Concept to teach
            complexity (float): Concept complexity (0-1)
            scaffolding_levels (int): Number of scaffolding levels. Default 3.
            
        Returns:
            list: Sequence of scaffolded prompts
        """
        
    def optimize_for_cognitive_load(self, prompt, target_load='medium'):
        """Optimize prompt for specific cognitive load.
        
        Args:
            prompt (str): Prompt to optimize
            target_load (str): Target cognitive load ('low', 'medium', 'high')
            
        Returns:
            str: Cognitive load optimized prompt
        """
```

### 6. Utilities (`prompt_theory/utils/`)

```
utils/
├── __init__.py
├── text_processing.py         # Text processing utilities
├── metrics.py                 # Metric calculation utilities
├── llm_interface.py           # LLM API interface
└── data_handling.py           # Data handling utilities
```

#### Key Functions in `utils/`

##### `text_processing.py`

```python
def tokenize(text, tokenizer=None):
    """Tokenize text using specified tokenizer.
    
    Args:
        text (str): Text to tokenize
        tokenizer (callable, optional): Tokenizer function
        
    Returns:
        list: Tokenized text
    """
    
def estimate_working_memory_load(text):
    """Estimate working memory load of text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Estimated working memory load (chunks)
    """
    
def segment_by_cognitive_chunks(text, max_chunk_size=4):
    """Segment text into cognitive chunks.
    
    Args:
        text (str): Text to segment
        max_chunk_size (int): Maximum chunk size (items). Default 4.
        
    Returns:
        list: Text segmented into cognitive chunks
    """
```

## Integration Examples

### Basic Prompt Optimization

```python
from prompt_theory.optimizers import PromptOptimizer
from prompt_theory.models import AttentionModel, RecursiveProcessor, DriftModel

# Initialize component models
attention_model = AttentionModel(recency_bias=0.3)
recursion_model = RecursiveProcessor(max_depth=3)
drift_model = DriftModel(drift_detection_threshold=0.2)

# Initialize optimizer with models
optimizer = PromptOptimizer(
    model="gpt-4",
    attention_model=attention_model,
    recursive_processor=recursion_model,
    drift_model=drift_model
)

# Define base prompt and task
base_prompt = "Explain how a transformer neural network works"
task = "educational"
context = {"audience": "undergraduate students", "prior_knowledge": "basic neural networks"}

# Optimize prompt
optimized_prompt = optimizer.optimize(
    base_prompt=base_prompt,
    task=task,
    context=context,
    constraints={"max_tokens": 800}
)

# Generate response with optimized prompt
response = optimizer.generate(optimized_prompt)
```

### Educational Application

```python
from prompt_theory.applications.education import EducationalPromptDesigner
from prompt_theory.optimizers import NeurobiologicalOptimizer

# Initialize neurobiological optimizer
optimizer = NeurobiologicalOptimizer(model="gpt-4")

# Initialize educational prompt designer
designer = EducationalPromptDesigner(optimizer=optimizer)

# Design instructional prompt
prompt = designer.design_instructional_prompt(
    topic="quantum computing",
    student_level="undergraduate physics",
    learning_objectives=[
        "Understand qubits and superposition",
        "Grasp quantum gates and operations",
        "Comprehend basic quantum algorithms"
    ]
)

# Design scaffolded learning sequence
scaffolded_prompts = designer.design_scaffolded_prompt(
    concept="Bayesian inference",
    complexity=0.7,
    scaffolding_levels=4
)
```

### Evaluation and Visualization

```python
from prompt_theory.evaluation import PromptEffectivenessEvaluator
from prompt_theory.visualization import AttentionVisualizer

# Initialize evaluator
evaluator = PromptEffectivenessEvaluator()

# Evaluate prompt effectiveness
effectiveness = evaluator.evaluate(
    prompt=optimized_prompt,
    task=task,
    context=context
)

# Initialize visualizer
visualizer = AttentionVisualizer()

# Visualize attention patterns
attention_weights = attention_model.allocate(
    context=["part1", "part2", "part3", "part4"],
    queries=["q1", "q2"],
    keys=["k1", "k2", "k3", "k4"],
    values=["v1", "v2", "v3", "v4"]
)

attention_figure = visualizer.plot_attention_map(
    attention_weights=attention_weights,
    labels=["Part 1", "Part 2", "Part 3", "Part 4"]
)
attention_figure.savefig("attention_map.png")
```

## Configuration System

The package includes a flexible configuration system that allows customization of models, optimizers, and applications:

```python
from prompt_theory import config

# Set global configuration
config.set_global({
    "default_model": "gpt-4",
    "attention": {
        "recency_bias": 0.3,
        "salience_weight": 0.5
    },
    "recursion": {
        "max_depth": 4,
        "collapse_threshold": 0.75
    }
})

# Get configuration value
model_name = config.get("default_model")

# Create context-specific configuration
with config.context({"recursion.max_depth": 6}):
    # Code in this block uses depth=6
    optimizer = PromptOptimizer()  # Uses depth=6
    
# Outside the context, original config is restored
optimizer = PromptOptimizer()  # Uses depth=4
```

## Extending the Package

The package is designed to be extensible. Users can create custom components by inheriting from the base classes:

```python
from prompt_theory.optimizers import PromptOptimizer
from prompt_theory.models import AttentionModel

class CustomAttentionModel(AttentionModel):
    """Custom attention model with domain-specific logic."""
    
    def __init__(self, domain_params, **kwargs):
        super().__init__(**kwargs)
        self.domain_params = domain_params
        
    def allocate(self, context, queries, keys, values):
        # Custom allocation logic
        allocation = super().allocate(context, queries, keys, values)
        # Domain-specific modifications
        # ...
        return allocation

class DomainSpecificOptimizer(PromptOptimizer):
    """Domain-specific prompt optimizer."""
    
    def __init__(self, domain, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        
    def domain_specific_optimization(self, prompt):
        # Implement domain-specific optimization
        # ...
        return optimized_prompt
```
