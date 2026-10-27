# LAB-2681 · Building Quantum-Capable AI Models

## 1 · Set up (once, about 1 minute)

Open this folder in a IBM BOB IDE and run the following command in the terminal inside folder `LAB-2681-starter-files`:

```bash
bash setup.sh
```

It creates a private Python 3.11 environment in `./.venv` (no admin rights needed), installs Mellea and
Qiskit, and checks that everything works. Wait for **"All set!"**.

## 2 · Run the lab

1. Open the folder `LAB-2681-starter-files` in IBM Bob IDE.
2. Open **`LAB-2681_quantum_data_pipeline.ipynb`**.
3. Click on "Select Kernel" -> "Python Environment" -> ".venv(3.11.6) (Python 3.11.6)"
4. If you don't see the installed python environment in step 3, try restarting the IBM Bob IDE.
5. Run the cells from top to bottom.

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
