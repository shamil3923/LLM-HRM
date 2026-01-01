"""
Example usage of the HRM + LLM hybrid solver.

This script demonstrates how to use the solver with actual LLM integration.
You'll need to implement call_llm in llm_interface.py first.
"""

from hybrid_solver import (
    solve,
    HRMCore,
    train_hrm,
    generate_synthetic_example,
    plan_to_hrm_input,
    compute_answer_from_plan
)
import torch


def main():
    """Main example workflow."""
    
    print("=" * 80)
    print("HRM + LLM Hybrid Solver - Example Usage")
    print("=" * 80)
    
    # Example 1: Train a new HRM model
    print("\n1. Training HRM Model")
    print("-" * 80)
    print("Training on 2000 synthetic examples...")
    
    model = HRMCore()
    train_hrm(model, num_steps=2000, print_every=500)
    
    # Save the trained model
    torch.save(model.state_dict(), "hrm_model.pt")
    print("\nModel saved to 'hrm_model.pt'")
    
    # Example 2: Test on synthetic data
    print("\n\n2. Testing HRM on Synthetic Data")
    print("-" * 80)
    
    model.eval()
    total_error = 0
    num_tests = 10
    
    for i in range(num_tests):
        plan, true_answer = generate_synthetic_example(max_depth=3)
        hrm_input = plan_to_hrm_input(plan)
        
        with torch.no_grad():
            predicted = model(hrm_input).item()
        
        error = abs(predicted - true_answer)
        total_error += error
        
        print(f"Test {i+1}: True={true_answer:.2f}, Predicted={predicted:.2f}, Error={error:.4f}")
    
    avg_error = total_error / num_tests
    print(f"\nAverage Error: {avg_error:.4f}")
    
    # Example 3: Solve real problems (requires LLM implementation)
    print("\n\n3. Solving Real Problems")
    print("-" * 80)
    print("NOTE: This requires implementing call_llm in llm_interface.py")
    print("      For now, this will raise NotImplementedError")
    
    problems = [
        "Alice has 15 pencils. She gives 7 to Bob. How many pencils does Alice have now?",
        "A rectangle has length 8 meters and width 5 meters. What is its area?",
        "Tom earns $20 per hour. He works 6 hours. How much does he earn in total?"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}: {problem}")
        
        try:
            result = solve(problem)
            print(f"Answer: {result['answer']}")
            print(f"Explanation: {result['explanation']}")
            print(f"Plan: {result['plan']}")
        except NotImplementedError:
            print("⚠️  Cannot solve: call_llm not implemented yet")
            print("   Please implement the LLM interface first")
            break
    
    # Example 4: Load existing model
    print("\n\n4. Loading Pre-trained Model")
    print("-" * 80)
    
    loaded_model = HRMCore()
    loaded_model.load_state_dict(torch.load("hrm_model.pt"))
    loaded_model.eval()
    print("Model loaded successfully from 'hrm_model.pt'")
    
    # Test loaded model
    plan, true_answer = generate_synthetic_example(max_depth=2)
    hrm_input = plan_to_hrm_input(plan)
    
    with torch.no_grad():
        predicted = loaded_model(hrm_input).item()
    
    print(f"Verification test: True={true_answer:.2f}, Predicted={predicted:.2f}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    
    print("\n\nNext Steps:")
    print("1. Implement call_llm in llm_interface.py with your LLM API")
    print("2. Test with real math word problems")
    print("3. Fine-tune the HRM model with more training steps if needed")
    print("4. Adjust MAX_ENTITIES and MAX_OPS if handling more complex problems")


if __name__ == "__main__":
    main()
