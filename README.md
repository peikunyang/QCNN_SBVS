# QCNN_SBVS
 Quantum Convolution for Structure-Based Virtual Screening

## Project Directory Structure
- **1_database/1_structure**: Contains occupancy data for training and testing.
- **1_database/2_pdk**: Contains binding free energy data for training and testing.
- **2_train**: Includes the training scripts.

> **Note:**  
> The data in this project is derived from PDBbind 2020, which includes a total of 19,443 entries. Due to storage limitations, only 10 sample entries are provided.

---
To set up the required environment, install the following dependencies:

1. **Install NumPy (repeat to ensure installation):**
   ```bash
  pip install numpy
  pip install pennylane

