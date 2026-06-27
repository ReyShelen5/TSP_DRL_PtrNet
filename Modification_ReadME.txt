# Modifications to TSP-DRL-PtrNet

This repository contains modifications to the original TSP-DRL-PtrNet implementation to support custom TSP instances and flexible coordinate inputs.

---

## Modified Files

### 1. `utils.py`

**Changes Made:**
- Modified the data generation process to support reading city coordinates from a CSV file.
- Added functionality to load user-defined TSP instances instead of generating random coordinates.
- Updated preprocessing to convert CSV data into tensors compatible with the model.

**Purpose:**
Allows the model to work with externally provided datasets and benchmark instances.

---

### 2. `train.py`

**Changes Made:**
- Updated the training pipeline to use custom datasets loaded through the modified data loader.
- Removed dependency on randomly generated city coordinates during training.

**Purpose:**
Enables training on user-defined TSP datasets.

---

### 3. `test.py`

**Changes Made:**
- Modified the testing pipeline to evaluate the model using custom CSV datasets.
- Updated input loading to accept externally supplied city coordinates.

**Purpose:**
Allows evaluation on benchmark or real-world TSP instances.

---

### 4. `config.py` *(if modified)*

**Changes Made:**
- Added configuration options for specifying custom dataset paths.
- Updated default parameters to support CSV-based input.

---

## Functional Changes

The following functionality has been added:

- Support for loading city coordinates from CSV files.
- Acceptance of user-defined TSP instances.
- Support for coordinates beyond the normalized range of `[0, 1]`.
- Compatibility with benchmark datasets for testing and evaluation.

---

## Unchanged Components

The following components remain unchanged from the original implementation:

- Pointer Network architecture
- Encoder and decoder
- Attention mechanism
- Reinforcement learning framework
- Reward computation
- Tour construction process
- Training algorithm

---

## Notes

These modifications only affect the input generation and data loading pipeline. The core learning algorithm and network architecture remain identical to the original implementation.