"""
Configuration file for the Quanvolutional Neural Network model.
"""

import torch

# --- Dataset Configuration ---
TRAIN_PATH = '/content/drive/MyDrive/set/train'
TEST_PATH = '/content/drive/MyDrive/set/test'
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
IMAGE_SIZE = 32 # 32x32 pixels

# --- Model Configuration ---
N_QUBITS = 4 # Number of qubits for the quanvolutional layer
CONV_KERNEL_SIZE = 3
FC1_OUTPUT = 64
# Calculate the output size after the Quanv and Conv layers
# Quanv output size: IMAGE_SIZE // 2 = 16
# Conv output size: (16 - KERNEL_SIZE) + 1 = 14
CONV_OUTPUT_SIZE = (IMAGE_SIZE // 2 - CONV_KERNEL_SIZE) + 1
FC1_INPUT = 16 * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE


# --- Training Configuration ---
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.1

# --- Quantum Device Configuration ---
# Use 'default.qubit' for CPU simulation
# Use 'lightning.gpu' for GPU simulation if pennylane-lightning-gpu is installed
QUANTUM_DEVICE = 'lightning.gpu'

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 