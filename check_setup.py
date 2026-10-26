"""Quick check that the lab environment works. Run by setup.sh, or on its own:  .venv/bin/python check_setup.py"""
import importlib
import sys

ok = True
print(f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 11):
    print("  ✗ Python 3.11+ is needed (Mellea requires it). Re-run: bash setup.sh"); ok = False

for mod in ["mellea", "litellm", "qiskit", "qiskit_aer", "qiskit_ibm_runtime", "flake8_qiskit_migration",
            "pytest", "matplotlib", "pandas", "ipykernel"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ✓ {mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {mod}: {e}"); ok = False

if ok:
    # The real pipeline end to end, on one prompt: Mellea's loop against the replay model + the unit tests.
    import logging
    logging.disable(logging.WARNING)
    import lab_helpers as H
    import replay_server
    from data_augmentation.core.llm_client import make_session_factory
    from data_augmentation.models import GenConfig

    url, _ = replay_server.start(delay=0.0)
    cfg = GenConfig(model="hosted_vllm/replay", loop_budget=5, max_tokens=4096)
    problems = H.load_jsonl("sample_training_data_20_input.jsonl")
    trace = H.ivr_trace(problems[2], 2, make_session_factory(url, "EMPTY", None, cfg), cfg)
    if trace["solved_at_turn"] == 2:
        print("  ✓ Mellea Instruct-Validate-Repair loop + unit tests: working")
    else:
        print(f"  ✗ pipeline self-test gave an unexpected result: {trace.get('skipped') or trace['solved_at_turn']}"); ok = False

print("\nAll set! Open LAB-2681_quantum_data_pipeline.ipynb in VS Code and pick the kernel"
      " 'LAB-2681 (Mellea + Qiskit)'." if ok else "\nSomething is missing: ask an instructor.")
sys.exit(0 if ok else 1)
