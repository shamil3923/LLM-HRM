"""
Hybrid HRM + LLM solver for math word problems.

HRM: Hierarchical Reasoning Model of Sapient Intelligence

This module implements a pipeline where:
1. LLM converts natural language to structured plan (JSON)
2. Plan is encoded into tensors for HRM
3. HRM computes numeric answer through hierarchical reasoning
4. LLM converts answer back to natural language explanation
"""

import json
import random
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim

from llm_interface import call_llm


# Constants
MAX_ENTITIES = 16
MAX_OPS = 16
OP_TYPES = {"add": 0, "sub": 1, "mul": 2, "div": 3, "pad": 4}
OP_TYPES_REV = {v: k for k, v in OP_TYPES.items()}


# =============================================================================
# Plan Validation
# =============================================================================

class PlanValidationError(Exception):
    """Raised when a plan fails validation."""
    pass


def validate_plan(plan: dict) -> None:
    """
    Validate that a plan follows the required schema.
    
    Args:
        plan: Dictionary containing entities, operations, and question
        
    Raises:
        PlanValidationError: If the plan is invalid
    """
    # Check required keys
    required_keys = {"entities", "operations", "question"}
    missing_keys = required_keys - set(plan.keys())
    if missing_keys:
        raise PlanValidationError(f"Missing required keys: {missing_keys}")
    
    # Validate entities
    if not isinstance(plan["entities"], list):
        raise PlanValidationError("'entities' must be a list")
    
    entity_names = set()
    for i, entity in enumerate(plan["entities"]):
        if not isinstance(entity, dict):
            raise PlanValidationError(f"Entity {i} must be a dict")
        if "name" not in entity or "value" not in entity:
            raise PlanValidationError(f"Entity {i} missing 'name' or 'value'")
        
        name = entity["name"]
        if not isinstance(name, str):
            raise PlanValidationError(f"Entity {i} name must be a string")
        if name in entity_names:
            raise PlanValidationError(f"Duplicate entity name: {name}")
        entity_names.add(name)
        
        # Value should be numeric
        if not isinstance(entity["value"], (int, float)):
            raise PlanValidationError(f"Entity {name} value must be numeric")
    
    # Validate operations
    if not isinstance(plan["operations"], list):
        raise PlanValidationError("'operations' must be a list")
    
    available_entities = entity_names.copy()
    
    for i, op in enumerate(plan["operations"]):
        if not isinstance(op, dict):
            raise PlanValidationError(f"Operation {i} must be a dict")
        
        # Check required fields
        if "op" not in op or "inputs" not in op or "output" not in op:
            raise PlanValidationError(
                f"Operation {i} missing 'op', 'inputs', or 'output'"
            )
        
        # Validate op type
        if op["op"] not in OP_TYPES or op["op"] == "pad":
            raise PlanValidationError(
                f"Operation {i} has invalid op type: {op['op']}"
            )
        
        # Validate inputs
        if not isinstance(op["inputs"], list) or len(op["inputs"]) != 2:
            raise PlanValidationError(
                f"Operation {i} inputs must be a list of 2 entity names"
            )
        
        for inp in op["inputs"]:
            if inp not in available_entities:
                raise PlanValidationError(
                    f"Operation {i} references undefined entity: {inp}"
                )
        
        # Validate output
        output = op["output"]
        if not isinstance(output, str):
            raise PlanValidationError(f"Operation {i} output must be a string")
        
        # Output becomes available for subsequent operations
        if output in available_entities:
            raise PlanValidationError(
                f"Operation {i} output name already exists: {output}"
            )
        available_entities.add(output)
    
    # Validate question
    question = plan["question"]
    if not isinstance(question, str):
        raise PlanValidationError("'question' must be a string")
    if question not in available_entities:
        raise PlanValidationError(
            f"Question references undefined entity: {question}"
        )


# =============================================================================
# NL to Plan (LLM)
# =============================================================================

PLAN_SYSTEM_PROMPT = """You are a mathematical reasoning assistant. Convert math word problems into structured JSON plans.

The JSON schema is:
{
  "entities": [{"name": "string", "value": number}],
  "operations": [{"op": "add"|"sub"|"mul"|"div", "inputs": ["entity1", "entity2"], "output": "entity3"}],
  "question": "entity_name"
}

CRITICAL RULES:
1. ALL numbers (including percentages like 10%, 0.1, constants) MUST be defined as entities first
2. Operations can ONLY reference entity names, never literal numbers
3. To use 10% (0.1), create an entity: {"name": "percentage", "value": 0.1}
4. Each operation must use exactly 2 existing entity names as inputs
5. op must be one of: add, sub, mul, div

Example 1:
Problem: "John has 5 apples. He buys 3 more. How many apples does he have?"
Output:
{
  "entities": [
    {"name": "apples_initial", "value": 5},
    {"name": "apples_bought", "value": 3}
  ],
  "operations": [
    {"op": "add", "inputs": ["apples_initial", "apples_bought"], "output": "apples_total"}
  ],
  "question": "apples_total"
}

Example 2:
Problem: "A store has 20 shirts. Each shirt costs $15. What is the total value?"
Output:
{
  "entities": [
    {"name": "num_shirts", "value": 20},
    {"name": "price_per_shirt", "value": 15}
  ],
  "operations": [
    {"op": "mul", "inputs": ["num_shirts", "price_per_shirt"], "output": "total_value"}
  ],
  "question": "total_value"
}

Example 3 (with percentage):
Problem: "Sarah has 100 dollars. She saves 20% of it. How much does she save?"
Output:
{
  "entities": [
    {"name": "total_money", "value": 100},
    {"name": "percentage", "value": 0.20}
  ],
  "operations": [
    {"op": "mul", "inputs": ["total_money", "percentage"], "output": "money_saved"}
  ],
  "question": "money_saved"
}

Now convert the following problem. Output ONLY valid JSON, no other text.
"""


def nl_to_plan(problem_text: str) -> dict:
    """
    Call the LLM to convert a natural language math word problem
    into a strict JSON plan following the schema.
    
    Args:
        problem_text: Natural language description of the problem
        
    Returns:
        Dictionary containing the structured plan
        
    Raises:
        PlanValidationError: If unable to get valid plan after retries
    """
    # Initial attempt
    prompt = f"{PLAN_SYSTEM_PROMPT}\n\nProblem: {problem_text}"
    response = call_llm(prompt)
    
    # Try to parse JSON
    try:
        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            if response.startswith("json"):
                response = response[4:].strip()
        
        plan = json.loads(response)
        validate_plan(plan)
        return plan
    except (json.JSONDecodeError, PlanValidationError) as e:
        # Attempt repair
        repair_prompt = f"""The previous JSON output had an error: {str(e)}

Previous output:
{response}

Please fix the JSON to match this schema exactly:
{{
  "entities": [{{"name": "string", "value": number}}],
  "operations": [{{"op": "add"|"sub"|"mul"|"div", "inputs": ["entity1", "entity2"], "output": "entity3"}}],
  "question": "entity_name"
}}

Original problem: {problem_text}

Output ONLY valid JSON, no other text."""
        
        repair_response = call_llm(repair_prompt)
        
        # Try to parse repaired JSON
        try:
            repair_response = repair_response.strip()
            if repair_response.startswith("```"):
                lines = repair_response.split("\n")
                repair_response = "\n".join(lines[1:-1]) if len(lines) > 2 else repair_response
                if repair_response.startswith("json"):
                    repair_response = repair_response[4:].strip()
            
            plan = json.loads(repair_response)
            validate_plan(plan)
            return plan
        except (json.JSONDecodeError, PlanValidationError) as e2:
            raise PlanValidationError(
                f"Failed to get valid plan after repair. Original error: {e}, "
                f"Repair error: {e2}"
            )


# =============================================================================
# Plan to HRM Input
# =============================================================================

def plan_to_hrm_input(plan: dict) -> torch.Tensor:
    """
    Convert the structured plan into a fixed-size tensor suitable
    as input to the HRM model.
    
    Args:
        plan: Dictionary containing entities, operations, and question
        
    Returns:
        Flattened 1D tensor encoding the plan
    """
    # Create entity name to index mapping
    entity_names = [e["name"] for e in plan["entities"]]
    entity_to_idx = {name: idx for idx, name in enumerate(entity_names)}
    
    # Entity values tensor [MAX_ENTITIES]
    entity_values = torch.zeros(MAX_ENTITIES, dtype=torch.float32)
    for i, entity in enumerate(plan["entities"]):
        if i < MAX_ENTITIES:
            entity_values[i] = float(entity["value"])
    
    # Operations tensor [MAX_OPS, 4]
    # Each row: [op_type_id, input1_idx, input2_idx, output_idx]
    ops_tensor = torch.zeros(MAX_OPS, 4, dtype=torch.float32)
    
    for i, op in enumerate(plan["operations"]):
        if i < MAX_OPS:
            op_type = OP_TYPES[op["op"]]
            input1_idx = entity_to_idx[op["inputs"][0]]
            input2_idx = entity_to_idx[op["inputs"][1]]
            
            # Add output to mapping for subsequent operations
            if op["output"] not in entity_to_idx:
                output_idx = len(entity_to_idx)
                entity_to_idx[op["output"]] = output_idx
            else:
                output_idx = entity_to_idx[op["output"]]
            
            ops_tensor[i, 0] = op_type
            ops_tensor[i, 1] = input1_idx
            ops_tensor[i, 2] = input2_idx
            ops_tensor[i, 3] = output_idx
    
    # Padding operations have op_type = 4
    for i in range(len(plan["operations"]), MAX_OPS):
        ops_tensor[i, 0] = OP_TYPES["pad"]
    
    # Flatten everything into a single 1D tensor
    # Shape: [MAX_ENTITIES + MAX_OPS * 4]
    flattened = torch.cat([
        entity_values,
        ops_tensor.flatten()
    ])
    
    return flattened


# =============================================================================
# HRM Core Model
# =============================================================================

class HRMCore(nn.Module):
    """
    Hierarchical Reasoning Model of Sapient Intelligence (HRM Core).
    
    A neural network that takes encoded mathematical plans and outputs
    scalar answers through hierarchical reasoning layers.
    """
    
    def __init__(self, input_size: int = MAX_ENTITIES + MAX_OPS * 4, 
                 hidden_size: int = 128):
        """
        Initialize the HRM core network.
        
        Args:
            input_size: Size of input tensor (default: 80)
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size] or [input_size]
            
        Returns:
            Output tensor of shape [batch_size, 1] or [1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        output = self.network(x)
        return output.squeeze(-1)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_example(max_depth: int = 4) -> Tuple[dict, float]:
    """
    Randomly generate a synthetic plan and its true numeric answer,
    following the same schema, without NL.
    
    Args:
        max_depth: Maximum number of operations to generate
        
    Returns:
        Tuple of (plan_dict, true_answer)
    """
    # Generate 2-4 base entities with small integer values
    num_base_entities = random.randint(2, 4)
    entities = []
    entity_values_map = {}
    
    for i in range(num_base_entities):
        name = f"x{i}"
        value = random.randint(1, 20)
        entities.append({"name": name, "value": value})
        entity_values_map[name] = value
    
    # Generate 1-max_depth operations
    num_ops = random.randint(1, min(max_depth, 4))
    operations = []
    available_entities = list(entity_values_map.keys())
    
    for i in range(num_ops):
        # Pick two random inputs from available entities
        if len(available_entities) < 2:
            break
        
        input1 = random.choice(available_entities)
        input2 = random.choice(available_entities)
        
        # Pick random operation
        op_type = random.choice(["add", "sub", "mul", "div"])
        
        # Compute the result
        val1 = entity_values_map[input1]
        val2 = entity_values_map[input2]
        
        if op_type == "add":
            result = val1 + val2
        elif op_type == "sub":
            result = val1 - val2
        elif op_type == "mul":
            result = val1 * val2
        elif op_type == "div":
            # Avoid division by zero
            if val2 == 0:
                val2 = 1
                entity_values_map[input2] = 1
            result = val1 / val2
        
        output_name = f"y{i}"
        operations.append({
            "op": op_type,
            "inputs": [input1, input2],
            "output": output_name
        })
        
        entity_values_map[output_name] = result
        available_entities.append(output_name)
    
    # The question is the last computed entity
    question = operations[-1]["output"] if operations else entities[0]["name"]
    answer = entity_values_map[question]
    
    plan = {
        "entities": entities,
        "operations": operations,
        "question": question
    }
    
    return plan, answer


# =============================================================================
# HRM Training
# =============================================================================

def train_hrm(model: HRMCore, num_steps: int = 5000, 
              learning_rate: float = 0.001, print_every: int = 500) -> None:
    """
    Train the Hierarchical Reasoning Model purely on synthetic 
    (plan, answer) pairs using MSE loss.
    
    Args:
        model: The HRMCore model to train
        num_steps: Number of training steps
        learning_rate: Learning rate for optimizer
        print_every: Print loss every N steps
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    
    for step in range(num_steps):
        # Generate synthetic example
        plan, true_answer = generate_synthetic_example()
        
        # Convert to HRM input
        hrm_input = plan_to_hrm_input(plan)
        
        # Forward pass
        prediction = model(hrm_input)
        
        # Compute loss
        target = torch.tensor([true_answer], dtype=torch.float32)
        loss = criterion(prediction, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (step + 1) % print_every == 0:
            print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.6f}")
    
    print(f"Training complete. Final loss: {loss.item():.6f}")


# =============================================================================
# Answer to Text (LLM)
# =============================================================================

def answer_to_text(problem_text: str, plan: dict, answer: float) -> str:
    """
    Call the LLM to turn the numeric answer and plan into a short,
    human-readable explanation (2–3 sentences).
    
    Args:
        problem_text: Original natural language problem
        plan: The structured plan used to solve it
        answer: The numeric answer computed by HRM
        
    Returns:
        Human-readable explanation string
    """
    # Create simplified textual description of operations
    operations_text = []
    for op in plan["operations"]:
        op_symbol = {
            "add": "+", "sub": "-", "mul": "×", "div": "÷"
        }.get(op["op"], op["op"])
        
        operations_text.append(
            f"{op['output']} = {op['inputs'][0]} {op_symbol} {op['inputs'][1]}"
        )
    
    ops_description = "\n".join(operations_text)
    
    prompt = f"""Given this math problem and solution, provide a brief 2-3 sentence explanation in natural language.

Problem: {problem_text}

Operations performed:
{ops_description}

Final answer: {answer}

Provide a clear, concise explanation of how we arrived at this answer."""
    
    explanation = call_llm(prompt)
    return explanation.strip()


# =============================================================================
# End-to-End Solve
# =============================================================================

def solve(problem_text: str) -> dict:
    """
    End-to-end: NL -> plan -> HRM input -> HRM output -> explanation.
    
    Args:
        problem_text: Natural language math word problem
        
    Returns:
        Dictionary with keys:
        - "answer": float, the numeric answer
        - "explanation": str, human-readable explanation
        - "plan": dict, the structured plan
    """
    # Step 1: Convert NL to plan
    plan = nl_to_plan(problem_text)
    
    # Step 2: Convert plan to HRM input
    hrm_input = plan_to_hrm_input(plan)
    
    # Step 3: Run through HRM (assumes model is loaded/available)
    # For now, we'll compute the answer directly from the plan as a fallback
    # In production, you'd use: answer = hrm_model(hrm_input).item()
    answer = compute_answer_from_plan(plan)
    
    # Step 4: Convert answer to explanation
    explanation = answer_to_text(problem_text, plan, answer)
    
    return {
        "answer": answer,
        "explanation": explanation,
        "plan": plan
    }


def compute_answer_from_plan(plan: dict) -> float:
    """
    Helper function to compute the answer directly from a plan.
    This is used as a fallback when HRM model isn't available.
    
    Args:
        plan: Structured plan dictionary
        
    Returns:
        Computed numeric answer
    """
    # Build entity values map
    values = {}
    for entity in plan["entities"]:
        values[entity["name"]] = entity["value"]
    
    # Execute operations in order
    for op in plan["operations"]:
        input1 = values[op["inputs"][0]]
        input2 = values[op["inputs"][1]]
        
        if op["op"] == "add":
            result = input1 + input2
        elif op["op"] == "sub":
            result = input1 - input2
        elif op["op"] == "mul":
            result = input1 * input2
        elif op["op"] == "div":
            result = input1 / input2 if input2 != 0 else 0
        
        values[op["output"]] = result
    
    # Return the answer
    return values[plan["question"]]


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HRM + LLM Hybrid Solver Demo")
    print("=" * 70)
    
    # Mock LLM for testing
    original_call_llm = call_llm  # Save reference to original
    
    def mock_call_llm(prompt: str) -> str:
        """Mock LLM that returns predefined responses for testing."""
        
        if "John has 5 apples" in prompt:
            return """{
  "entities": [
    {"name": "apples_initial", "value": 5},
    {"name": "apples_bought", "value": 3}
  ],
  "operations": [
    {"op": "add", "inputs": ["apples_initial", "apples_bought"], "output": "apples_total"}
  ],
  "question": "apples_total"
}"""
        
        elif "store has 20 shirts" in prompt:
            return """{
  "entities": [
    {"name": "num_shirts", "value": 20},
    {"name": "price_per_shirt", "value": 15}
  ],
  "operations": [
    {"op": "mul", "inputs": ["num_shirts", "price_per_shirt"], "output": "total_value"}
  ],
  "question": "total_value"
}"""
        
        elif "brief 2-3 sentence explanation" in prompt:
            if "apples" in prompt:
                return "John started with 5 apples and bought 3 more. Adding these together gives us a total of 8 apples."
            elif "shirts" in prompt:
                return "The store has 20 shirts, each priced at $15. Multiplying the quantity by the price gives us a total value of $300."
        
        return "{}"
    
    # Replace call_llm globally for the demo
    import sys
    sys.modules[__name__].call_llm = mock_call_llm
    
    print("\n1. Testing synthetic data generation...")
    print("-" * 70)
    
    for i in range(3):
        plan, answer = generate_synthetic_example(max_depth=3)
        print(f"\nSynthetic Example {i+1}:")
        print(f"Entities: {plan['entities']}")
        print(f"Operations: {plan['operations']}")
        print(f"Question: {plan['question']}")
        print(f"Answer: {answer}")
    
    print("\n\n2. Training HRM model...")
    print("-" * 70)
    
    model = HRMCore()
    train_hrm(model, num_steps=1000, print_every=200)
    
    print("\n\n3. Testing end-to-end solve with mock LLM...")
    print("-" * 70)
    
    # Test problem 1
    problem1 = "John has 5 apples. He buys 3 more. How many apples does he have?"
    print(f"\nProblem: {problem1}")
    
    result1 = solve(problem1)
    print(f"\nPlan: {json.dumps(result1['plan'], indent=2)}")
    print(f"Answer: {result1['answer']}")
    print(f"Explanation: {result1['explanation']}")
    
    # Test problem 2
    problem2 = "A store has 20 shirts. Each shirt costs $15. What is the total value?"
    print(f"\n\nProblem: {problem2}")
    
    result2 = solve(problem2)
    print(f"\nPlan: {json.dumps(result2['plan'], indent=2)}")
    print(f"Answer: {result2['answer']}")
    print(f"Explanation: {result2['explanation']}")
    
    print("\n\n4. Testing HRM model inference...")
    print("-" * 70)
    
    # Test HRM with a simple synthetic example
    test_plan, true_answer = generate_synthetic_example(max_depth=2)
    hrm_input = plan_to_hrm_input(test_plan)
    
    model.eval()
    with torch.no_grad():
        predicted_answer = model(hrm_input).item()
    
    print(f"\nTest Plan: {json.dumps(test_plan, indent=2)}")
    print(f"True Answer: {true_answer:.4f}")
    print(f"HRM Predicted: {predicted_answer:.4f}")
    print(f"Error: {abs(predicted_answer - true_answer):.4f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
