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

base folder
	|
	--- model.py
	|
	--- spatial.py
	|
	--- dataset.py
	|
	--- train.py 
	|
	--- data
	    |
	    --- trn_X.npy
	    |
	    --- trn_Y.npy
	    |
	    --- trn_T.npy
	    |
	    --- normalization.npy

```


## Usage

## Example

