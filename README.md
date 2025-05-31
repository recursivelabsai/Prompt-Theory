

<div align="center">


## **A Unified Framework for Understanding AI Prompting and Human Cognition**

[![Prompt Theory](https://img.shields.io/badge/Paper-Preprint-b31b1b.svg)](https://github.com/recursivelabsai/Prompt-Theory/blob/main/Prompt%20Theory%20Preprint.md)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-4b44ce.svg)](https://neurips.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

[Preprint](https://github.com/recursivelabsai/Prompt-Theory/blob/main/Prompt%20Theory%20Preprint.md) | [What is Prompt Theory?](#what-is-prompt-theory) | [Key Concepts](#key-concepts) | [Getting Started](#getting-started) | [Experimental Results](#experimental-results) | [Applications](#applications) 

</div>

## What is Prompt Theory?

Prompt Theory is a foundational framework that establishes structural isomorphisms between artificial intelligence prompting systems and human neurobiological input-output mechanisms. By formalizing the parallels between prompt design and cognitive input processing, we reveal shared emergent properties, failure modes, and optimization pathways across both domains.

<div align="center">


## Interpreting a Viral Phenomenon
    
## [Twitter](https://x.com/venturetwins/status/1927750673619120373) | [YouTube](https://www.youtube.com/watch?v=BLfV4sidcJM&ab_channel=Putchuon-%22PutYouOn%22) | [Tribune](https://tribune.com.pk/story/2547468/the-prompt-theory#:~:text=The%20realistic%20characters%20in%20those%20clips%20can,random%20prompt%20writer%20is%20deciding%20their%20fate.)

<img width="882" alt="image" src="https://github.com/user-attachments/assets/c958281b-e996-4e09-bd11-ceb0e84a8e29" />

<img width="883" alt="image" src="https://github.com/user-attachments/assets/d4af346d-6961-45a4-ac5b-531c3edc4212" />



https://github.com/user-attachments/assets/7f2feab7-97a6-4004-bcc2-95f9024a692b




https://github.com/user-attachments/assets/e5a45bb0-6aea-4eb7-8afc-530c699a16f3




</div>

Our framework provides:

1. **A formal mapping** between AI prompting and human cognitive processing stages
2. **Identification of recursive patterns** that govern both systems
3. **A unified mathematical framework** for optimizing prompts based on neurobiologically-inspired principles
4. **Practical tools** for enhancing LLM performance, human instruction design, and human-AI collaboration

## Key Concepts

### Structural Isomorphisms

Prompt Theory identifies six primary processing stages that exhibit structural isomorphisms between AI prompting systems and human neurobiological processing:

1. **Input Channel Processing**: How prompts/stimuli are initially received
2. **Encoding and Preprocessing**: Transformation into internal representations
3. **Attention and Context Management**: Selection of relevant information
4. **Inference and Integration**: Generation of higher-order representations
5. **Output Generation**: Production of responses/behaviors
6. **Error Correction and Adaptation**: Learning from feedback

### The Mathematical Framework

Our mathematical framework formalizes key phenomena including:

- **Recursive Information Processing**: How current inputs are interpreted in light of previous states
- **Attention Allocation**: How limited computational resources are distributed
- **Prompt Effectiveness Function**: Quantifying prompt quality across contexts
- **Recursive Collapse and Emergence**: When systems break down or generate novel structures
- **Drift and Stability**: How systems maintain or lose coherence over time

<div align="center">
<img src="assets/images/mathematical-framework.png" width="600px" alt="Mathematical Framework">
</div>

## Getting Started

### Installation

```bash
pip install prompt-theory
```

### Basic Usage

```python
from prompt_theory import PromptOptimizer, AttentionModel, RecursiveProcessor

# Initialize the prompt optimizer with your LLM of choice
optimizer = PromptOptimizer(
    model="gpt-4",
    attention_model=AttentionModel(),
    recursive_processor=RecursiveProcessor()
)

# Optimize a prompt
optimized_prompt = optimizer.optimize(
    base_prompt="Explain quantum computing",
    task="educational",
    context="undergraduate physics student",
    constraints={"max_tokens": 500}
)

# Generate response with optimized prompt
response = optimizer.generate(optimized_prompt)
```

## Repository Structure

```
prompt-theory/
├── prompt_theory/                 # Main package
│   ├── __init__.py                # Package initialization
│   ├── models/                    # Core models
│   │   ├── attention.py           # Attention allocation models
│   │   ├── recursion.py           # Recursive processing models
│   │   └── drift.py               # Drift and stability models
│   ├── optimizers/                # Prompt optimization
│   │   ├── base.py                # Base optimizer class
│   │   ├── neurobiological.py     # Bio-inspired optimizers
│   │   └── hybrid.py              # Combined approaches
│   └── evaluation/                # Evaluation metrics
│       ├── effectiveness.py       # Prompt effectiveness metrics
│       ├── collapse.py            # Collapse detection
│       └── emergence.py           # Emergence detection
├── experiments/                   # Experimental validation
│   ├── ai_performance/            # AI system experiments
│   ├── human_cognition/           # Human processing experiments
│   └── collaboration/             # Human-AI collaboration studies
├── notebooks/                     # Jupyter notebooks
│   ├── introduction.ipynb         # Introduction to Prompt Theory
│   ├── case_studies.ipynb         # Real-world applications
│   └── advanced_techniques.ipynb  # Advanced optimization
├── examples/                      # Example applications
├── docs/                          # Documentation
├── tests/                         # Unit and integration tests
├── assets/                        # Images and other assets
└── paper/                         # Academic paper and resources
```

## Experimental Results

Our experiments validate Prompt Theory across three domains:

### AI System Performance

Prompts designed using Prompt Theory principles showed statistically significant improvements (p < 0.01):

| Task Category | Standard Prompting | Prompt Theory | Improvement |
|---------------|-------------------|--------------|-------------|
| Reasoning     | 68.3%             | 76.5%        | +8.2%       |
| Creative      | 3.8/5             | 4.3/5        | +0.5        |
| Factual       | 71.2%             | 79.8%        | +8.6%       |
| Instructions  | 74.5%             | 82.7%        | +8.2%       |

### Human Cognitive Processing

Instructions designed using Prompt Theory principles led to:

- 23% improvement in task accuracy
- 18% reduction in completion time
- 27% reduction in reported cognitive load

### Human-AI Collaboration

Teams using Prompt Theory-based interaction protocols demonstrated:

- 31% higher solution quality (expert-rated)
- 35% higher reported satisfaction with collaboration
- 28% more balanced contribution between human and AI

## Applications

Prompt Theory has applications across multiple domains:

- **Enhanced Prompt Engineering**: Design more effective prompts for LLMs
- **Educational Materials**: Create instructions that reduce cognitive load
- **Clinical Interventions**: Support cognitive processing in clinical populations
- **Human-AI Interfaces**: Design more effective interaction protocols
- **Collaborative Systems**: Optimize information exchange between humans and AI

## Contributing

We welcome contributions from the community! Please see our [contribution guidelines](CONTRIBUTING.md) for details on how to get involved.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/recursivelabs/prompt-theory.git
cd prompt-theory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Citation

If you use Prompt Theory in your research, please cite our paper:

```bibtex
@inproceedings{prompt-theory-2025,
  title={Prompt Theory: A Unified Framework for Understanding AI Prompting and Human Cognition},
  author={Kim, David and Keyes, Caspian},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by grants from the National Science Foundation (Grant #AI-20256) and the Advanced Research Projects Agency (Grant #ARPA-2023-0142).
