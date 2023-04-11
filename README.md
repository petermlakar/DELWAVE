<p align="center">
    <img src="images/DELWAVE_logo.png" alt="DELWAVE logo" width="300px">
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
	    --- training_wind_field.npy
	    |
	    --- training_waves.npy
	    |
	    --- training_time.npy
	    |
	    --- normalization.npy

```

The training and test data files can be found on [Zenodo]{https://doi.org/10.5281/zenodo.7816888}.

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

After the training data has be acquired and places in the approriate subdirectory as described in the [Setup](#Setup) section,
the train.py script can be used to fir the DELWAVE model.

```console
python3 train.py <training dataset name> <number of time steps> <path to base folder>>

# <training dataset name>: The name of the station data which is to be used for training. Options include AA, MB, GD, OB, OB2, OB3.
#                          If WHOLE is supplied instead the training is conducted on all stations at the same time. This applies if the provided dataset is used.
#
# <number of time steps>:  The number of consequitve wind field time steps used for rgeression. 
#
# <path to base folder>:   Path to the base folder where the remaining required script files are located.
```

The train.py script creates a folder named DELWAVEv1.0_results which contains the trained model.

## Example

Usage example when training on all provided stations, with 11 time steps used for regression.

``` console
python3 train.py WHOLE 11 /path/to/DELWAVE/base/folder
```

After the training is complete you can find a new folder named DELWAVEv1.0 inside the base folder.
This folder contains the trained DELWAVE model.
To produce the test results for any of the stations used during training, execute:

``` console
python3 test.py <AA or GD or MB or ...> DELWAVEv1.0 /path/to/DELWAVE/base/folder
```

The test.py script will produce the DELWAVE prediction statistics for the test dataset of choice, while also outputing the prediction arrays
for the mean wave period, significant wave height, and mean wave direction.
