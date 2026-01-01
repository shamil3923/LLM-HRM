#!/bin/bash
# Quick commands for HRM + LLM project

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HRM + LLM Hybrid Solver - Quick Commands${NC}"
echo -e "${BLUE}========================================${NC}"

# Activate virtual environment
echo -e "\n${GREEN}1. Activating virtual environment...${NC}"
source venv/bin/activate

# Show menu
echo -e "\n${YELLOW}What would you like to do?${NC}"
echo "1) Run demo (quick test)"
echo "2) Train better model (20k steps, ~5 minutes)"
echo "3) Train production model (50k steps, ~15 minutes)"
echo "4) Test existing model"
echo "5) Run full example"
echo "6) Exit"

read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running demo...${NC}"
        python demo_quick.py
        ;;
    2)
        echo -e "\n${GREEN}Training better model (20k steps)...${NC}"
        python -c "
from hybrid_solver import HRMCore, train_hrm
import torch

print('Training HRM with 20,000 steps...')
model = HRMCore()
train_hrm(model, num_steps=20000, print_every=2000)
torch.save(model.state_dict(), 'hrm_20k.pt')
print('\n✓ Model saved to hrm_20k.pt')
"
        ;;
    3)
        echo -e "\n${GREEN}Training production model (50k steps)...${NC}"
        python -c "
from hybrid_solver import HRMCore, train_hrm
import torch

print('Training production HRM with 50,000 steps...')
model = HRMCore(hidden_size=256)
train_hrm(model, num_steps=50000, print_every=5000)
torch.save(model.state_dict(), 'hrm_production.pt')
print('\n✓ Production model saved to hrm_production.pt')
"
        ;;
    4)
        echo -e "\n${GREEN}Testing existing model...${NC}"
        python -c "
from hybrid_solver import HRMCore, generate_synthetic_example, plan_to_hrm_input
import torch

model = HRMCore()
model.load_state_dict(torch.load('hrm_trained.pt'))
model.eval()

print('Testing on 10 synthetic examples:\n')
total_error = 0
for i in range(10):
    plan, true_ans = generate_synthetic_example(max_depth=3)
    hrm_input = plan_to_hrm_input(plan)
    with torch.no_grad():
        pred_ans = model(hrm_input).item()
    error = abs(true_ans - pred_ans)
    total_error += error
    print(f'Test {i+1:2d}: True={true_ans:8.2f}, Pred={pred_ans:8.2f}, Error={error:6.2f}')

avg_error = total_error / 10
print(f'\nAverage Error: {avg_error:.2f}')
"
        ;;
    5)
        echo -e "\n${GREEN}Running example.py...${NC}"
        python example.py
        ;;
    6)
        echo -e "\n${YELLOW}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${YELLOW}Invalid choice${NC}"
        ;;
esac

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${BLUE}========================================${NC}"
