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
* **Operating system**: Tested in Ubuntu 24.04
* **PyTorch version**: 2.9.1

Install other required packages:

```bash
pip install -r requirements.txt
```

## Basic Usage

Run the following command to train the model with a specific problem setup:

```bash
python train_il.py \
   --env_is_gym=0 \
   --algorithm=demodice \
   --env_id=Hopper \
   --xml_path=./env/hopper_target.xml \
  --dataset_file_names="['target_hopper_5_3091.0554050953983_1000.0.npz', 'target_hopper_400_3075.603734081881_997.5925.npz', 'target_Hopper-v3_random_50.npz']" \
  --load_hdf5_dataset=0 \
```


* `algorithm`: Corresponds to the specific method used.
* `env_id`: Corresponds to a specific environment task.
* `xml_path`: Defines the specific XML configuration for the environment.
* `dataset_file_names`: Lists the dataset files used for training.

