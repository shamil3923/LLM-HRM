# Setup Guide

## Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/hrm-llm-solver.git
cd hrm-llm-solver
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up OpenAI API key:**

Option A - Environment variable (recommended):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Option B - Edit `llm_interface.py`:
```python
client = OpenAI(api_key="your-api-key-here")
```

5. **Run the demo:**
```bash
python demo_real.py
```

## Troubleshooting

### Import errors
Make sure you activated the virtual environment:
```bash
source venv/bin/activate
```

### API key errors
Verify your OpenAI API key is set correctly:
```bash
echo $OPENAI_API_KEY
```

### Model not found
The pre-trained model `hrm_trained.pt` should be included. If missing, train a new one:
```bash
./quick_commands.sh
# Select option 2: Train model
```

## Usage Examples

### Simple problem:
```python
from hybrid_solver import solve

result = solve("Tom has 15 apples. He gives 7 to Mary. How many apples does Tom have left?")
print(result['answer'])  # 8.0
```

### Complex problem:
```python
problem = "A shop has 37 red pens and 45 blue pens. On Tuesday, they sell 12 red pens and 8 blue pens. On Wednesday, 10% of the remaining pens are damaged and removed. How many pens are left?"
result = solve(problem)
print(result['answer'])  # 78.2
```
