"""
Test complex multi-step math problem with HRM + OpenAI
"""

from hybrid_solver import solve, compute_answer_from_plan
import json

problem = """A shop has 37 red pens and 45 blue pens on Monday.
On Tuesday, it sells 18 red pens and 12 blue pens, and receives a shipment of 25 red pens and 5 blue pens.
On Wednesday morning, the manager removes 10% of the remaining blue pens because they are damaged.
How many pens are left in total after the damaged blue pens are removed?"""

print("="*80)
print("Testing Complex Multi-Step Problem")
print("="*80)
print(f"\n📝 Problem:\n{problem}\n")
print("="*80)

# Manual calculation for verification
print("\n🧮 Manual Calculation (Ground Truth):")
print("-"*80)
red_monday = 37
blue_monday = 45
print(f"Monday: {red_monday} red, {blue_monday} blue")

# Tuesday transactions
red_after_tuesday = red_monday - 18 + 25  # 37 - 18 + 25 = 44
blue_after_tuesday = blue_monday - 12 + 5  # 45 - 12 + 5 = 38
print(f"Tuesday: Sell 18 red, 12 blue; Receive 25 red, 5 blue")
print(f"After Tuesday: {red_after_tuesday} red, {blue_after_tuesday} blue")

# Wednesday - remove 10% of blue pens
blue_damaged = blue_after_tuesday * 0.10  # 38 * 0.10 = 3.8
blue_after_wednesday = blue_after_tuesday - blue_damaged  # 38 - 3.8 = 34.2
print(f"Wednesday: Remove 10% of blue ({blue_damaged:.1f} pens)")
print(f"After Wednesday: {red_after_tuesday} red, {blue_after_wednesday:.1f} blue")

total_ground_truth = red_after_tuesday + blue_after_wednesday  # 44 + 34.2 = 78.2
print(f"\n✓ Ground Truth Total: {total_ground_truth} pens")

print("\n" + "="*80)
print("🤖 Testing HRM + OpenAI System")
print("="*80)

try:
    # Solve using the hybrid system
    result = solve(problem)
    
    print("\n📋 Generated Plan:")
    print("-"*80)
    print(json.dumps(result['plan'], indent=2))
    
    # Compute ground truth from plan
    plan_ground_truth = compute_answer_from_plan(result['plan'])
    
    print("\n" + "="*80)
    print("📊 Results Comparison")
    print("="*80)
    print(f"Expected Answer (manual):     {total_ground_truth}")
    print(f"Plan Execution (from plan):   {plan_ground_truth}")
    print(f"HRM Prediction:               {result['answer']:.2f}")
    
    error_vs_expected = abs(result['answer'] - total_ground_truth)
    error_vs_plan = abs(result['answer'] - plan_ground_truth)
    
    print(f"\nError vs Expected:            {error_vs_expected:.2f}")
    print(f"Error vs Plan Execution:      {error_vs_plan:.2f}")
    
    print("\n💬 AI Explanation:")
    print("-"*80)
    print(result['explanation'])
    
    print("\n" + "="*80)
    print("🎯 Analysis")
    print("="*80)
    
    # Check if the plan is correct
    if abs(plan_ground_truth - total_ground_truth) < 0.1:
        print("✅ OpenAI generated a CORRECT plan!")
        if error_vs_plan < 5:
            print("✅ HRM executed the plan with good accuracy!")
            print("\n🎉 SUCCESS! System handled the complex problem well!")
        else:
            print("⚠️  HRM prediction has some error (needs more training)")
            print(f"   Consider training longer: ./quick_commands.sh (option 2)")
    else:
        print("⚠️  OpenAI's plan may have issues")
        print(f"   Plan produces: {plan_ground_truth}")
        print(f"   Expected: {total_ground_truth}")
        print("\n   Possible issues:")
        print("   1. LLM struggled with multi-step reasoning")
        print("   2. Plan may be missing some operations")
        print("   3. Might need better prompting")
    
    # Check plan complexity
    num_entities = len(result['plan']['entities'])
    num_operations = len(result['plan']['operations'])
    
    print(f"\n📈 Plan Complexity:")
    print(f"   Entities: {num_entities}")
    print(f"   Operations: {num_operations}")
    
    if num_operations >= 5:
        print("   ✓ Plan has good complexity for this problem")
    else:
        print("   ⚠️  Plan might be too simple for this problem")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 This problem might be too complex. Possible issues:")
    print("   1. Too many entities/operations for current limits")
    print("   2. Percentage calculations might confuse the system")
    print("   3. Multi-day tracking requires careful plan construction")

print("\n" + "="*80)
