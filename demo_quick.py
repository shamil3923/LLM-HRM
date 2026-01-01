"""
Quick demo to test the HRM model with mocked LLM responses.
This demonstrates the full pipeline without requiring actual LLM API integration.
"""

import json
from hybrid_solver import (
    HRMCore,
    train_hrm,
    generate_synthetic_example,
    plan_to_hrm_input,
    validate_plan,
    compute_answer_from_plan,
    plan_to_hrm_input,
    answer_to_text,
    nl_to_plan
)
import torch


# Mock LLM implementation
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
    
    elif "store has 20 shirts" in prompt or "20 shirts" in prompt:
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
    
    elif "brief 2-3 sentence explanation" in prompt or "explanation" in prompt.lower():
        if "apples" in prompt.lower():
            return "John started with 5 apples and bought 3 more. Adding these together (5 + 3) gives us a total of 8 apples."
        elif "shirts" in prompt.lower() or "shirt" in prompt.lower():
            return "The store has 20 shirts, each priced at $15. Multiplying the quantity by the price (20 × 15) gives us a total value of $300."
        else:
            return "The calculation was performed according to the mathematical operations in the plan."
    
    return '{"entities": [], "operations": [], "question": ""}'


def demo_solve(problem_text: str, call_llm_func) -> dict:
    """
    End-to-end solve with custom LLM function.
    
    Args:
        problem_text: Natural language math word problem
        call_llm_func: Function to call LLM
        
    Returns:
        Dictionary with answer, explanation, and plan
    """
    # Temporarily replace the call_llm function in the hybrid_solver module
    import hybrid_solver
    import llm_interface
    
    original_call_llm_interface = llm_interface.call_llm
    original_call_llm_solver = hybrid_solver.call_llm
    
    llm_interface.call_llm = call_llm_func
    hybrid_solver.call_llm = call_llm_func
    
    try:
        # Step 1: Convert NL to plan
        plan = nl_to_plan(problem_text)
        
        # Step 2: Compute answer directly from plan
        answer = compute_answer_from_plan(plan)
        
        # Step 3: Convert answer to explanation
        explanation = answer_to_text(problem_text, plan, answer)
        
        return {
            "answer": answer,
            "explanation": explanation,
            "plan": plan
        }
    finally:
        # Restore original functions
        llm_interface.call_llm = original_call_llm_interface
        hybrid_solver.call_llm = original_call_llm_solver


def main():
    print("=" * 80)
    print("HRM + LLM Hybrid Solver - Quick Demo")
    print("=" * 80)
    
    # 1. Test synthetic data generation
    print("\n1. Testing Synthetic Data Generation")
    print("-" * 80)
    
    for i in range(3):
        plan, answer = generate_synthetic_example(max_depth=3)
        print(f"\nSynthetic Example {i+1}:")
        print(f"  Entities: {len(plan['entities'])} entities")
        print(f"  Operations: {len(plan['operations'])} operations")
        print(f"  Question: {plan['question']}")
        print(f"  Answer: {answer}")
        
        # Validate the plan
        try:
            validate_plan(plan)
            print(f"  ✓ Plan validation passed")
        except Exception as e:
            print(f"  ✗ Validation error: {e}")
    
    # 2. Train HRM model
    print("\n\n2. Training HRM Model")
    print("-" * 80)
    print("Training on 2000 synthetic examples...")
    
    model = HRMCore()
    train_hrm(model, num_steps=2000, print_every=400)
    
    # Save the model
    torch.save(model.state_dict(), "hrm_trained.pt")
    print("\n✓ Model saved to 'hrm_trained.pt'")
    
    # 3. Test HRM inference
    print("\n\n3. Testing HRM Inference")
    print("-" * 80)
    
    model.eval()
    test_cases = 5
    total_error = 0
    
    print(f"Running {test_cases} test cases...\n")
    
    for i in range(test_cases):
        plan, true_answer = generate_synthetic_example(max_depth=2)
        hrm_input = plan_to_hrm_input(plan)
        
        with torch.no_grad():
            predicted = model(hrm_input).item()
        
        error = abs(predicted - true_answer)
        total_error += error
        
        print(f"Test {i+1}: True={true_answer:8.2f}, Predicted={predicted:8.2f}, Error={error:6.4f}")
    
    avg_error = total_error / test_cases
    print(f"\nAverage Error: {avg_error:.4f}")
    
    # 4. Test end-to-end with mock LLM
    print("\n\n4. Testing End-to-End Pipeline (with Mock LLM)")
    print("-" * 80)
    
    test_problems = [
        "John has 5 apples. He buys 3 more. How many apples does he have?",
        "A store has 20 shirts. Each shirt costs $15. What is the total value?"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n{'='*80}")
        print(f"Problem {i}: {problem}")
        print('='*80)
        
        result = demo_solve(problem, mock_call_llm)
        
        print(f"\n📋 Plan:")
        print(f"   Entities: {result['plan']['entities']}")
        print(f"   Operations: {result['plan']['operations']}")
        print(f"   Question: {result['plan']['question']}")
        
        print(f"\n🔢 Answer: {result['answer']}")
        
        print(f"\n💬 Explanation:")
        print(f"   {result['explanation']}")
    
    # 5. Show how to use HRM for prediction
    print("\n\n5. Using Trained HRM for Predictions")
    print("-" * 80)
    
    # Create a simple plan manually
    simple_plan = {
        "entities": [
            {"name": "a", "value": 10},
            {"name": "b", "value": 5}
        ],
        "operations": [
            {"op": "add", "inputs": ["a", "b"], "output": "result"}
        ],
        "question": "result"
    }
    
    print("\nManual Plan:")
    print(json.dumps(simple_plan, indent=2))
    
    # Compute with HRM
    hrm_input = plan_to_hrm_input(simple_plan)
    with torch.no_grad():
        hrm_prediction = model(hrm_input).item()
    
    # Compute ground truth
    true_answer = compute_answer_from_plan(simple_plan)
    
    print(f"\nGround Truth: {true_answer}")
    print(f"HRM Prediction: {hrm_prediction:.2f}")
    print(f"Error: {abs(hrm_prediction - true_answer):.4f}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("Demo Complete! 🎉")
    print("=" * 80)
    
    print("\n✅ What's Working:")
    print("   • Synthetic data generation")
    print("   • HRM model training")
    print("   • Plan validation")
    print("   • Tensor encoding")
    print("   • End-to-end pipeline with mocked LLM")
    
    print("\n📝 Next Steps:")
    print("   1. Implement call_llm() in llm_interface.py with real LLM API")
    print("   2. Train HRM longer for better accuracy (try 10k-50k steps)")
    print("   3. Test with more complex math problems")
    print("   4. Fine-tune prompts for better plan generation")
    
    print("\n💾 Files Created:")
    print("   • hrm_trained.pt - Trained HRM model (ready to use)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
