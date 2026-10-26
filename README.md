# LAB-2681 · Building Quantum-Capable AI Models

## 1 · Set up (once, about 1 minute)

Open this folder in a IBM BOB IDE and run the following command in the terminal:

```bash
bash setup.sh
```

It creates a private Python 3.11 environment in `./.venv` (no admin rights needed), installs Mellea and
Qiskit, and checks that everything works. Wait for **"All set!"**.

## 2 · Run the lab

1. Open **`LAB-2681_quantum_data_pipeline.ipynb`**.
2. Click on "Select Kernel" and install the required package.
3. Choose the kernel **"LAB-2681 (Mellea + Qiskit)"**: click *Select Kernel* (top right) → *Jupyter Kernel…*.
4. Run the cells from top to bottom.

## What's in this folder

| | |
|---|---|
| `LAB-2681_quantum_data_pipeline.ipynb` | The lab notebook |
| `setup.sh`, `requirements.txt`, `check_setup.py` | Environment setup and self-test |
| `sample_training_data_20_input.jsonl` | The 20 Qiskit prompts, each with its unit tests |
| `data_augmentation/` | The data-generation pipeline (Mellea IVR, correction and fill-in-the-blank stages) |
| `lab_helpers.py` | Display and plotting helpers for the notebook |
| `replay_server.py` | Plays back the recorded model answers |
| `data/` | Recorded model answers and training results |
