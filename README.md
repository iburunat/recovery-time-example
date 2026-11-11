# recovery-time-example
Estimating post-exercise recovery time: a worked example

## Run

This example uses simulated data only; it does not use real participant or Garmin data.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python recovery_time_example.py
```

The script prints model evaluation results and saves `recovery_time_example.png` in the current directory.
