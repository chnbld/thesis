###Real world anomaly detection in surveillance videos

###Environment
The implementation is tested using:
- Keras version 1.1.0
- Theano 1.0.5
- Python 3.8.5
- Debian 10
For me, this environment is built on docker.

###Training and evaluation
Please, create an output weight folder
Run `regular-train.py` to train and evaluate model 
The AUC result will be saved in `AUC.txt`

###Test custom data
Run `regular-test.py` to test custom data