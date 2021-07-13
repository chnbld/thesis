###Multiple Instance Learning for Child Abuse Detection based on Self-Distillation
####Environment
The implementation is tested using:
- Torch version 1.5.0
- Python 3.6.9
- NVIDIA TITAN RTX(GPU)
- Ubuntu 18.04
I built this environment on docker.
To create docker image, please check the URL `https://ngc.nvidia.com/catalog/containers/nvidia:pytorch`
The command is something like: `docker run --gpus all -it --ipc=host nvcr.io/nvidia/pytorch:20.03-py3`
I used `nvcr.io/nvidia/pytorch:20.03-py3` image,
The command param `--ipc=host` is important.

####Train and evaluation
Run `main.py` to train model.
The result AUC will be saved in AUC.txt.

####Test custom data
Change name of `option-test.py` to `option.py` and `test_10crop-test.py` to `test_10crop.py`
Run `main-test.py` to test custom data.

