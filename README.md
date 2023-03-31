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


### Train model with minimal configuration

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

The dataset files in the subfolder data can be obtained at:

### Custom model training loop

The DELWAVE model architecture can be accessed by including the model definition from model.py.
```python

from model import Model
from dataset import Databank, Dataset

# We initialize th DELWAVE model
delwave = Model(time_steps = 11)

# ... Prepare training dataset

X = # ... wind field input
Y = # ... wave attributes at station
S = # ... spatial encoding at station (as defined in spatial.py)
N = # ... Means and standard deviations for individual wave attributes at station and whole wind field

databank = Databank(X, Y, S, time_steps = 11, normalize = N, station_indices = [0], cuda = True)
dataset = Dataset(databank, databank.indices, batch_size = 256, importance = True)

# ... Fit model

```

## Usage

## Example

