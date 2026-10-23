# ICU Vision Monitor

Contactless heart-rate and respiratory-rate monitoring using RGB video and RealSense depth. This project is a research prototype, not a medical device.

**Repository:** [github.com/nijakaruppusamy25-sys/icu-monitor](https://github.com/nijakaruppusamy25-sys/icu-monitor)

## Setup

Requires Python 3.10 or later.

```bash
git clone https://github.com/nijakaruppusamy25-sys/icu-monitor.git
cd icu-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run an evaluation

```bash
python scripts/evaluate_rr_depth.py
python scripts/evaluate_hr_pos.py
```

The respiratory evaluator reports mean absolute error for paced trials. The heart-rate evaluators report estimates only; reference heart-rate data is not currently included.
