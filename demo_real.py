"""
Real HRM + LLM Demo using OpenAI API
Solves actual math word problems using OpenAI for language and HRM for computation
"""

from hybrid_solver import (
    HRMCore, 
    train_hrm,
    solve,
    compute_answer_from_plan
)
import torch

def main():
    """Run real OpenAI + HRM demo."""
    print("\n" + "="*80)
    print("HRM + LLM Hybrid Solver - Real OpenAI Demo")
    print("="*80)
    
    # Step 1: Load or train HRM model
    print("\n" + "="*80)
    print("Step 1: Loading HRM Model")
    print("="*80)
    
    model = HRMCore()
    
    try:
        model.load_state_dict(torch.load('hrm_trained.pt'))
        print("✓ Loaded existing model from hrm_trained.pt")
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        train_hrm(model, num_steps=2000, print_every=400)
        torch.save(model.state_dict(), 'hrm_trained.pt')
        print("✓ Model trained and saved to hrm_trained.pt")
    
    # Step 2: Test problems
    print("\n" + "="*80)
    print("Step 2: Solving Math Word Problems with Real OpenAI")
    print("="*80)
    
    problems = [
        "John has 5 apples. He buys 3 more apples. How many apples does John have in total?",
        "Sarah has 20 dollars. She spends 8 dollars on lunch. How much money does she have left?",
        "A box contains 15 chocolates. Tom eats 4 chocolates. How many chocolates remain in the box?",
        "Lisa reads 12 pages per day. After 3 days, how many pages has she read in total?",
    ]
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"Problem {i}/{len(problems)}")
        print('='*80)
        print(f"❓ {problem}\n")
        
        try:
            result = solve(problem)
            
            # Compute ground truth for verification
            ground_truth = compute_answer_from_plan(result['plan'])
            
            print(f"✓ Plan Generated:")
            print(f"  • Entities: {result['plan']['entities']}")
            print(f"  • Operations: {result['plan']['operations']}")
            print(f"  • Question: {result['plan']['question']}")
            
            print(f"\n🔢 Ground Truth Answer: {ground_truth}")
            print(f"🧠 HRM Predicted Answer: {result['answer']:.2f}")
            print(f"📊 Error: {abs(ground_truth - result['answer']):.2f}")
            
            print(f"\n💬 Explanation:")
            print(f"   {result['explanation']}")
            
            results.append((problem, result, True, ground_truth))
            
        except Exception as e:
            print(f"✗ Error solving problem: {e}")
            import traceback
            traceback.print_exc()
            results.append((problem, None, False, None))
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = sum(1 for _, _, success, _ in results if success)
    print(f"\n✓ Successfully solved: {successful}/{len(problems)} problems")
    
    if successful > 0:
        total_error = 0
        count = 0
        for _, result, success, truth in results:
            if success and result:
                error = abs(truth - result['answer'])
                total_error += error
                count += 1
        
        if count > 0:
            avg_error = total_error / count
            print(f"📊 Average HRM Error: {avg_error:.2f}")
            
            if avg_error > 10:
                print(f"\n💡 Tip: Your HRM needs more training!")
                print(f"   Run: ./quick_commands.sh and select option 2 (20k steps)")
            elif avg_error > 2:
                print(f"\n💡 HRM is doing OK, but can be improved with more training")
            else:
                print(f"\n🎉 HRM accuracy is excellent!")
    
    if successful == len(problems):
        print("\n" + "🎉"*20)
        print("SUCCESS! Your HRM + OpenAI hybrid system is working perfectly!")
        print("🎉"*20)
    else:
        print(f"\n⚠️ {len(problems) - successful} problems failed.")
        print("Check the errors above for details.")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Train HRM longer: ./quick_commands.sh (option 2 or 3)")
    print("2. Test with your own problems")
    print("3. Integrate into your application")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
