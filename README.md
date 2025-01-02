# QCNN_SBVS
 Quantum Convolution for Structure-Based Virtual Screening
# Occupancy and Binding Free Energy Prediction

This project aims to use PyTorch to train models for predicting occupancy and binding free energy.

## Project Directory Structure
- **1_database/1_structure**: Contains occupancy data for training and testing.
- **1_database/2_pdk**: Contains binding free energy data for training and testing.
- **2_train**: Includes the training scripts.

> **Note:**  
> The data in this project is derived from PDBbind 2020, which includes a total of 19,443 entries. Due to storage limitations, only 10 sample entries are provided.

---

## Installation Guide

### 1. Install PyTorch
Install PyTorch based on your operating system and hardware architecture. Detailed instructions are available on the [PyTorch official website](https://pytorch.org/get-started/locally/). For example, to install the GPU version, use the following command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
