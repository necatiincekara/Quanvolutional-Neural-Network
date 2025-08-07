#!/bin/bash

# Setup script for M4 Mac Mini development environment
# Optimized for Apple Silicon and local development

echo "================================================"
echo "Setting up Quantum ML Environment for M4 Mac"
echo "================================================"

# Check if running on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is designed for macOS"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo "✓ Apple Silicon M4 detected"
else
    echo "Warning: Not running on Apple Silicon"
fi

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.10 via Homebrew
echo "Installing Python 3.10..."
brew install python@3.10

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv_quantum

# Activate environment
source venv_quantum/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with MPS support
echo "Installing PyTorch with Metal Performance Shaders support..."
pip install torch torchvision torchaudio

# Install PennyLane with CPU optimization
echo "Installing PennyLane for CPU..."
pip install pennylane pennylane-lightning

# Install other requirements
echo "Installing other dependencies..."
pip install numpy scikit-learn tqdm PyYAML gradio tensorboard opencv-python

# Create local configuration
echo "Creating local configuration..."
cat > src/config_local.py << 'EOF'
"""
Local Mac M4 configuration for development and testing
"""
import torch
import os

# Dataset paths (update these to your local paths)
TRAIN_PATH = './data/sample_train'  # Small sample for testing
TEST_PATH = './data/sample_test'    # Small sample for testing

# Model parameters
TAGS = {
    '01': 'elif', '02': 'be', '03': 'te', '04': 'se', '05': 'cim', 
    '06': 'ha', '07': 'hı', '08': 'dal', '09': 'zel', '10': 'ra', 
    '11': 'ze', '12': 'sin', '13': 'şın', '14': 'şad', '15': 'dad', 
    '16': 'tı', '17': 'zı', '18': 'ayn', '19': 'ğayn', '20': 'fe', 
    '21': 'kaf', '22': 'kef', '23': 'lam', '24': 'mim', '25': 'nun', 
    '26': 'he', '27': 'vav', '28': 'ye', '29': 'pe', '30': 'çim', 
    '31': 'je', '32': 'gef', '33': 'nef', '34': '1', '35': '2', 
    '36': '3', '37': '4', '38': '5', '39': '6', '40': '7', 
    '41': '8', '42': '9', '43': '0', '44': 'lamelif'
}
NUM_CLASSES = len(TAGS)
IMAGE_SIZE = 32

# Quantum configuration - CPU optimized for Mac
N_QUBITS = 4
QUANTUM_DEVICE = 'lightning.qubit'  # CPU-optimized, not GPU

# Training configuration - reduced for local testing
BATCH_SIZE = 32  # Smaller batch for M4 Mac
NUM_EPOCHS = 5   # Quick testing epochs
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.1

# Device configuration - Use MPS for PyTorch operations
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Reproducibility
RANDOM_SEED = 42

print(f"Mac M4 Config Loaded: Device={DEVICE}, Quantum={QUANTUM_DEVICE}")
EOF

# Create sample data directory
echo "Creating sample data directory..."
mkdir -p data/sample_train
mkdir -p data/sample_test

# Create test script
echo "Creating test script..."
cat > test_mac_setup.py << 'EOF'
#!/usr/bin/env python
"""Test script to verify Mac M4 setup"""

import sys
import torch
import pennylane as qml

def test_setup():
    print("="*50)
    print("Testing Mac M4 Setup")
    print("="*50)
    
    # Test PyTorch
    print("\n1. PyTorch Configuration:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"   MPS device: {torch.device('mps')}")
    
    # Test PennyLane
    print("\n2. PennyLane Configuration:")
    print(f"   PennyLane version: {qml.__version__}")
    
    # Test quantum device
    try:
        dev = qml.device('lightning.qubit', wires=4)
        print(f"   Quantum device: ✓ lightning.qubit available")
    except:
        print(f"   Quantum device: ✗ lightning.qubit not available")
    
    # Test import of project modules
    print("\n3. Project Modules:")
    try:
        from src.config_local import DEVICE, QUANTUM_DEVICE
        print(f"   Config loaded: Device={DEVICE}, Quantum={QUANTUM_DEVICE}")
        print(f"   ✓ Local configuration working")
    except ImportError as e:
        print(f"   ✗ Error loading config: {e}")
    
    # Performance test
    print("\n4. Performance Test:")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(32, 1, 32, 32).to(device)
        print(f"   Created tensor on MPS: shape={x.shape}")
        print(f"   ✓ MPS acceleration available")
    else:
        print(f"   ✗ MPS not available, using CPU")
    
    print("\n" + "="*50)
    print("Setup test complete!")
    print("="*50)

if __name__ == "__main__":
    test_setup()
EOF

chmod +x test_mac_setup.py

# Create quick training script for Mac
echo "Creating quick training script..."
cat > train_local_mac.py << 'EOF'
#!/usr/bin/env python
"""Quick training script for Mac M4 local development"""

import torch
import sys
import os

# Use local config
os.environ['USE_LOCAL_CONFIG'] = '1'

# Import with local config
from src.config_local import *
from src.trainable_quantum_model import create_enhanced_model
from src.enhanced_training import EnhancedTrainer

def quick_train(epochs=2):
    """Quick training for testing on Mac"""
    print(f"Starting quick training on {DEVICE}")
    
    # Create model
    model = create_enhanced_model(circuit_type='hardware_efficient')
    
    # Create small dummy dataset for testing
    train_data = torch.randn(100, 1, 32, 32)
    train_labels = torch.randint(0, NUM_CLASSES, (100,))
    
    val_data = torch.randn(20, 1, 32, 32)
    val_labels = torch.randint(0, NUM_CLASSES, (20,))
    
    # Simple training loop
    device = torch.device(DEVICE)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        for i in range(0, len(train_data), BATCH_SIZE):
            batch_x = train_data[i:i+BATCH_SIZE].to(device)
            batch_y = train_labels[i:i+BATCH_SIZE].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Batch {i//BATCH_SIZE+1}, Loss: {loss.item():.4f}")
    
    print("Quick training complete!")
    return model

if __name__ == "__main__":
    quick_train()
EOF

chmod +x train_local_mac.py

echo ""
echo "================================================"
echo "✅ Mac M4 Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv_quantum/bin/activate"
echo "2. Test setup: python test_mac_setup.py"
echo "3. Quick train test: python train_local_mac.py"
echo ""
echo "For full training, use Google Colab with the provided notebook."
echo "================================================"