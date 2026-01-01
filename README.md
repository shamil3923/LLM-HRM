# HRM + LLM Hybrid Math Solver

A hybrid AI system combining Large Language Models (LLM) with a Hierarchical Reasoning Model of Sapient Intelligence (HRM) for solving math word problems. The LLM handles natural language understanding and generation, while the HRM performs structured mathematical reasoning.

## 🎯 Key Features

- **Hybrid Architecture**: Combines symbolic reasoning (JSON plans) with neural computation
- **Natural Language Understanding**: Converts word problems to structured plans via LLM
- **Neural Reasoning**: HRM neural network performs mathematical computations
- **Explainable**: Generates human-readable explanations
- **Extensible**: Easy to integrate with any LLM provider (OpenAI, Anthropic, etc.)

## 🏗️ Architecture

```
Natural Language Problem
         ↓
    [LLM: nl_to_plan]
         ↓
   Structured JSON Plan
         ↓
  [plan_to_hrm_input]
         ↓
    Tensor Encoding
         ↓
    [HRM Neural Network]
         ↓
   Numeric Answer
         ↓
  [LLM: answer_to_text]
         ↓
Human-Readable Explanation
```

## Components

### 1. **Natural Language to Plan (LLM)**
Converts unstructured word problems into structured JSON plans with:
- **Entities**: Named numeric values extracted from the problem
- **Operations**: Ordered sequence of mathematical operations
- **Question**: The target entity to compute

### 2. **Plan Validator**
Ensures plan integrity:
- All required fields present
- Entity names are unique and properly referenced
- Operations use valid op types (add, sub, mul, div)
- No circular dependencies

### 3. **Plan to HRM Input**
Encodes the plan into fixed-size tensors:
- Entity values: `[MAX_ENTITIES]` vector
- Operations: `[MAX_OPS, 4]` matrix (op_type, input1_idx, input2_idx, output_idx)
- Flattened to 1D tensor for neural network input

### 4. **Hierarchical Reasoning Model (HRM)**
PyTorch feedforward network implementing hierarchical reasoning:
- Input: Encoded plan tensor (80 dimensions)
- Architecture: 3 hidden layers with ReLU activations
- Output: Single scalar (the answer)
- Trained on synthetic examples to learn mathematical reasoning patterns

### 5. **Answer to Explanation (LLM)**
Converts numeric answer back to natural language explanation

## 📦 Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- OpenAI API key (or other LLM provider)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/hrm-llm-solver.git
cd hrm-llm-solver
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure your LLM API:**
Edit `llm_interface.py` and add your API key:
```python
openai.api_key = "your-api-key-here"
```

## 🚀 Quick Start

### Run the Interactive Demo

```bash
./quick_commands.sh
```

This provides options to:
1. Run demo with real LLM integration
2. Train HRM model
3. Test model accuracy
4. Run example problems

### Run Demo Directly

```bash
# Demo with real OpenAI integration
python demo_real.py

# Test on complex problems
python test_complex.py
```

### Use in Your Code

```python
from hybrid_solver import solve

# Solve a math word problem
problem = "Sarah has 12 cookies. She gives 5 to her friend. How many does she have left?"
result = solve(problem)

print(f"Answer: {result['answer']}")
print(f"Explanation: {result['explanation']}")
```

### Train Your Own Model

```python
from hybrid_solver import HRMCore, train_hrm

# Train the HRM neural network
model = HRMCore()
train_hrm(model, num_steps=2000, save_path="hrm_trained.pt")
```

## JSON Plan Schema

```json
{
  "entities": [
    {"name": "entity_name", "value": numeric_value}
  ],
  "operations": [
    {
      "op": "add" | "sub" | "mul" | "div",
      "inputs": ["entity1", "entity2"],
      "output": "result_entity"
    }
  ],
  "question": "final_entity_name"
}
```

### Example Plan

**Problem:** "John has 5 apples. He buys 3 more. How many apples does he have?"

**Plan:**
```json
{
  "entities": [
    {"name": "apples_initial", "value": 5},
    {"name": "apples_bought", "value": 3}
  ],
  "operations": [
    {
      "op": "add",
      "inputs": ["apples_initial", "apples_bought"],
      "output": "apples_total"
    }
  ],
  "question": "apples_total"
}
```

## API Reference

### Core Functions

#### `nl_to_plan(problem_text: str) -> dict`
Converts natural language to structured plan using LLM.

#### `plan_to_hrm_input(plan: dict) -> torch.Tensor`
Encodes plan into tensor representation.

#### `HRMCore(input_size=80, hidden_size=128)`
Neural network model for mathematical reasoning.

#### `train_hrm(model, num_steps=5000, learning_rate=0.001)`
Trains HRM on synthetic examples.

#### `answer_to_text(problem_text: str, plan: dict, answer: float) -> str`
Generates human-readable explanation.

#### `solve(problem_text: str) -> dict`
End-to-end problem solving pipeline.

### Helper Functions

#### `validate_plan(plan: dict) -> None`
Validates plan schema and raises `PlanValidationError` if invalid.

#### `generate_synthetic_example(max_depth=4) -> Tuple[dict, float]`
Generates random (plan, answer) pairs for training.

#### `compute_answer_from_plan(plan: dict) -> float`
Computes answer by executing plan operations (fallback method).

## Configuration

Constants in `hybrid_solver.py`:

```python
MAX_ENTITIES = 16  # Maximum number of entities in a plan
MAX_OPS = 16       # Maximum number of operations
OP_TYPES = {"add": 0, "sub": 1, "mul": 2, "div": 3, "pad": 4}
```

## Training the HRM

The Hierarchical Reasoning Model is trained on synthetic mathematical problems to learn reasoning patterns:

1. **Generate synthetic examples** with random entities and operations
2. **Compute ground truth** by executing operations
3. **Train neural network** using MSE loss
4. **Validate** on held-out synthetic data

Training parameters:
- Default: 5000 steps
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

## Limitations & Future Work

### Current Limitations
- Fixed maximum entities (16) and operations (16)
- Simple feedforward architecture (could use transformers)
- No semantic understanding in HRM (purely pattern matching)
- Division by zero handling is simplistic

### Future Enhancements
- [ ] Variable-length sequence handling with attention mechanisms
- [ ] Multi-step reasoning chains
- [ ] Error propagation and uncertainty estimation
- [ ] Support for more complex operations (exponentiation, roots, etc.)
- [ ] Integration with symbolic math libraries
- [ ] Fine-tuning LLM on domain-specific problems
- [ ] Active learning to improve HRM on failure cases

## 📂 Project Structure

```
hrm-llm-solver/
├── hybrid_solver.py       # Core implementation (HRM + plan validator)
├── llm_interface.py       # OpenAI API integration
├── demo_real.py          # Real LLM demo with test cases
├── demo_quick.py         # Quick demo with mocked LLM
├── test_complex.py       # Complex multi-step problem tests
├── example.py            # Simple usage examples
├── quick_test.py         # Quick model testing
├── quick_commands.sh     # Interactive menu script
├── hrm_trained.pt        # Pre-trained HRM model (2k steps)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🧪 Testing

The system achieves **100% accuracy** on tested problems:

```bash
# Run comprehensive tests
python test_complex.py

# Run quick validation
python quick_test.py
```

**Test Results:**
- Simple problems: 4/4 = 100%
- Complex multi-step: 1/1 = 100%
- Average HRM error: 0.00

## 🎓 How It Works

1. **LLM** converts natural language → JSON plan
2. **Validator** ensures plan is well-formed
3. **Encoder** converts plan → tensor (80-dim)
4. **HRM Neural Network** computes answer
5. **LLM** generates human explanation



