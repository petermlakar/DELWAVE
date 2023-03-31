<p align="center">
    <img src="images/DELWAVE_logo_new_new_new_new.png" alt="DELWAVE logo" width="300px">
</p>


DEep Learning WAVe Emulating model (DELWAVE) is a convolutional neural network model designed for emulating the numerical surface ocean wave model.

## Requirements

DELWAVE v1.0 requires the external libraries PyTorch and Numpy.
These can be installed by running the following command in a terminal on Linux:

```conole
pip3 install numpy==1.22.4 torch==1.13.1
```

## Setup

To train the DELWAVE model with minimal source file configuration the following folder structure is required:

```
#   ARGUMENTS: <training dataset name>: The name of the station data which is to be used for training. Options include AA, MB, GD, OB, OB2, OB3.
#                                       If WHOLE is supplied instead the training is conducted on all stations at the same time.
#              <number of time steps>:  The number of consequitve wind field time steps used for rgeression. 
#              <path to base folder>:   Path to the base folder where the remaining required script files are located. Expected base folder structure:
#                               
#                                           base folder
#                                               |
#                                               --- model.py
#                                               |
#                                               --- spatial.py
#                                               |
#                                               --- dataset.py
#                                               |
#                                               --- train.py 
#                                               |
#                                               --- data
#                                                    |
#                                                    --- trn_X.npy
#                                                    |
#                                                    --- trn_Y.npy
#                                                    |
#                                                    --- trn_T.npy
#                                                    |
#                                                    --- normalization.npy
```

## Usage

## Example

