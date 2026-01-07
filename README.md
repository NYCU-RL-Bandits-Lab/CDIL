# Semi-Supervised Cross-Domain Imitation Learning

## Code Structure

The codebase is organized as follows:
* `train_il.py` : Entry point for running experiments.
* `agent/` : Contains implementations of various imitation learning algorithms.
* `env/` : Includes MuJoCo `.xml` assets required for the target domain.
* `flow_model/` : Stores pre-trained flow models for each environment.
* `pretrained_models/` : Contains pre-trained DemoDICE models from the source domain.
* `utils/` : Utility functions and tools required by the methods.

## Requirements

* **Python version**: Tested in Python 3.10.19
* **Operating system**: Tested in Ubuntu 20.04
* **PyTorch version**: 2.9.1

Install other required packages:

```bash
pip install -r requirements.txt
```
