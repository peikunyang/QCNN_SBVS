# QCNN_SBVS
 Quantum Convolution for Structure-Based Virtual Screening

## Project Directory Structure
- **1_database/1_structure**: Contains occupancy data for training and testing.
- **1_database/2_pdk**: Contains binding free energy data for training and testing.
- **2_train**: Includes the training scripts.

> **Note:**  
> The data in this project is derived from PDBbind 2020, which includes a total of 19,443 entries. Due to storage limitations, only 10 sample entries are provided.

---

## Installation Guide

To set up the required environment, install the following dependencies:

```bash
# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install NumPy
pip install numpy
pip install numpy

# Install PennyLane
pip install pennylane



