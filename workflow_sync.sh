#!/bin/bash

# Hybrid Workflow Sync Script
# Seamlessly sync between Mac development and Colab training

# Configuration
REPO_DIR="$HOME/Desktop/Quanvolutional-Neural-Network"
DRIVE_DIR="$HOME/Library/CloudStorage/GoogleDrive-your-email@gmail.com/My Drive"
GITHUB_REPO="https://github.com/YOUR_USERNAME/Quanvolutional-Neural-Network.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Quantum ML Hybrid Workflow Manager${NC}"
echo -e "${GREEN}================================================${NC}"

# Function to sync to GitHub
sync_to_github() {
    echo -e "${YELLOW}Syncing to GitHub...${NC}"
    cd "$REPO_DIR"
    
    # Check for changes
    if [[ -n $(git status -s) ]]; then
        git add -A
        echo "Enter commit message:"
        read commit_msg
        git commit -m "$commit_msg"
        git push origin main
        echo -e "${GREEN}✓ Pushed to GitHub${NC}"
    else
        echo -e "${YELLOW}No changes to commit${NC}"
    fi
}

# Function to sync from GitHub
sync_from_github() {
    echo -e "${YELLOW}Pulling from GitHub...${NC}"
    cd "$REPO_DIR"
    git pull origin main
    echo -e "${GREEN}✓ Updated from GitHub${NC}"
}

# Function to prepare for Colab
prepare_for_colab() {
    echo -e "${YELLOW}Preparing for Colab training...${NC}"
    
    # Create lightweight config for Colab
    cat > "$REPO_DIR/src/config_colab.py" << 'EOF'
"""Colab-specific configuration"""
import torch

# Paths for Google Drive
TRAIN_PATH = '/content/drive/MyDrive/set/train'
TEST_PATH = '/content/drive/MyDrive/set/test'

# Optimized for GPU training
BATCH_SIZE = 256  # Larger batch for A100
NUM_EPOCHS = 100  # Full training
LEARNING_RATE = 0.001

# Quantum device
QUANTUM_DEVICE = 'lightning.gpu'
DEVICE = "cuda"

print("Colab configuration loaded")
EOF
    
    # Sync to GitHub
    sync_to_github
    
    echo -e "${GREEN}✓ Ready for Colab training${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Open colab_training_optimized.ipynb in Google Colab"
    echo "2. Run all cells to start training"
    echo "3. Monitor progress via tensorboard"
}

# Function to download results from Drive
download_results() {
    echo -e "${YELLOW}Downloading results from Google Drive...${NC}"
    
    # Create results directory
    mkdir -p "$REPO_DIR/results"
    
    # Copy from Google Drive (if mounted)
    if [ -d "$DRIVE_DIR/quantum_checkpoints" ]; then
        cp -r "$DRIVE_DIR/quantum_checkpoints" "$REPO_DIR/results/"
        echo -e "${GREEN}✓ Downloaded checkpoints${NC}"
    fi
    
    if [ -d "$DRIVE_DIR/quantum_experiments" ]; then
        cp -r "$DRIVE_DIR/quantum_experiments" "$REPO_DIR/results/"
        echo -e "${GREEN}✓ Downloaded experiments${NC}"
    fi
    
    echo -e "${GREEN}Results saved to $REPO_DIR/results/${NC}"
}

# Function to run local test
run_local_test() {
    echo -e "${YELLOW}Running local test on Mac...${NC}"
    cd "$REPO_DIR"
    
    # Activate virtual environment
    source venv_quantum/bin/activate
    
    # Run quick test
    python << 'EOF'
import torch
from src.trainable_quantum_model import create_enhanced_model

# Test on MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = create_enhanced_model().to(device)

# Quick forward pass
x = torch.randn(4, 1, 32, 32).to(device)
output = model(x)

print(f"✓ Model working on {device}")
print(f"  Output shape: {output.shape}")
EOF
    
    deactivate
}

# Function to compare performance
compare_performance() {
    echo -e "${YELLOW}Performance Comparison${NC}"
    echo -e "${YELLOW}-------------------${NC}"
    
    cat << 'EOF'
    
    Platform Comparison for Your Setup:
    
    ┌─────────────────┬──────────────┬─────────────────┐
    │ Task            │ M4 Mac Mini  │ Colab Pro A100  │
    ├─────────────────┼──────────────┼─────────────────┤
    │ Development     │ ⭐⭐⭐⭐⭐    │ ⭐⭐            │
    │ Debugging       │ ⭐⭐⭐⭐⭐    │ ⭐⭐            │
    │ Quick Tests     │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐          │
    │ Full Training   │ ⭐⭐         │ ⭐⭐⭐⭐⭐       │
    │ Quantum Sim     │ ⭐⭐         │ ⭐⭐⭐⭐⭐       │
    │ Batch Size      │ 64           │ 256             │
    │ Epoch Time      │ ~3-4 hours   │ ~1 hour         │
    │ Availability    │ 24/7         │ Session limits  │
    └─────────────────┴──────────────┴─────────────────┘
    
    Recommendation: Use Mac for development, Colab for training
    
EOF
}

# Main menu
show_menu() {
    echo ""
    echo "Select action:"
    echo "1) Run local test (Mac)"
    echo "2) Prepare for Colab training" 
    echo "3) Sync to GitHub"
    echo "4) Pull from GitHub"
    echo "5) Download results from Drive"
    echo "6) Compare platform performance"
    echo "7) Full workflow (1→2→3)"
    echo "8) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-8]: " choice
    
    case $choice in
        1)
            run_local_test
            ;;
        2)
            prepare_for_colab
            ;;
        3)
            sync_to_github
            ;;
        4)
            sync_from_github
            ;;
        5)
            download_results
            ;;
        6)
            compare_performance
            ;;
        7)
            echo -e "${GREEN}Running full workflow...${NC}"
            run_local_test
            prepare_for_colab
            sync_to_github
            echo -e "${GREEN}✓ Full workflow complete${NC}"
            ;;
        8)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
done