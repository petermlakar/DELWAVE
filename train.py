import torch
import torch.nn as nn
import torch.jit as jit

import numpy as np

from datetime import datetime
import sys

from time import time
from datetime import datetime

from os import mkdir
from os.path import exists, join

#########################################################################

from model import Model

from dataset import Databank, Dataset
from spatial import spatial_map

#########################################################################

# DELWAVE training script
# Usage example: python3 train.py <training dataset name> <number of time steps> <path to base folder> <OPTIONAL: generated file postfix>
#                python3 train.py WHOLE 11 /home/DELWAVE_base TEST_EXAMPLE                
#
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

def main():

    CUDA = torch.cuda.is_available()

    if len(sys.argv) < 4:
        print("python3 train.py <training dataset name> <number of time steps> <path to base folder>")
        exit()

    dataset_name = sys.argv[1]

    dmap = {"AA": 0,
            "MB": 1,
            "GD": 2,
            "OB": 3,
            "OB2": 4,
            "OB3": 5}

    #########################################################################

    learning_rate = 1e-5
    time_steps = int(sys.argv[2])
    batch_size = 2 # Batch size is multiplied by the number of stations
    epochs = 5000
    tolerance = 5000

    print("Reading dataset...")

    BASE = sys.argv[3]
    BASE_TR = join(BASE, "data")

    X = np.load(join(BASE_TR, "training_wind_field.npy"))
    Y = np.load(join(BASE_TR, "training_waves.npy"))
    T = np.load(join(BASE_TR, "training_time.npy"))

    S_AA = spatial_map("AA")
    S_MB = spatial_map("MB")
    S_GD = spatial_map("GD")
    S_OB = spatial_map("OB")
    S_OB2 = spatial_map("OB2")
    S_OB3 = spatial_map("OB3")

    station_indices = []
    if dataset_name == "AA":
        S = np.expand_dims(S_AA, axis = 0)
        station_indices.append(0)
    elif dataset_name == "MB":
        S = np.expand_dims(S_MB, axis = 0)
        station_indices.append(1)
    elif dataset_name == "GD":
        S = np.expand_dims(S_GD, axis = 0)
        station_indices.append(2)
    elif dataset_name == "OB":
        S = np.expand_dims(S_OB, axis = 0)
        station_indices.append(3)
    elif dataset_name == "OB2":
        S = np.expand_dims(S_OB2, axis = 0)
        station_indices.append(4)
    elif dataset_name == "OB3":
         S = np.expand_dims(S_OB3, axis = 0)
         station_indices.append(5)
    else:
        S = np.stack([S_AA, S_MB, S_GD, S_OB, S_OB2, S_OB3], axis = 0)
        station_indices = [0, 1, 2, 3, 4, 5]

    if dataset_name != "WHOLE":
        Y = np.expand_dims(Y[dmap[dataset_name]], axis = 0)

    #########################################################################

    databank = Databank(X, Y, S, time_steps = time_steps, normalize = np.load(join(BASE_TR, "normalization.npy")), station_indices = station_indices, cuda = CUDA)

    #########################################################################
    # Split off last 20 percent

    split_ratio = int(databank.indices.size*0.80)
    trn_indices = np.copy(databank.indices[:split_ratio])
    vld_indices = np.copy(databank.indices[split_ratio + time_steps - 2:])

    #########################################################################

    dataset_trn = Dataset(databank, trn_indices, batch_size = batch_size, importance = True,  cuda = False) # Importance sampling must be on at init for latel toggling
    dataset_vld = Dataset(databank, vld_indices, batch_size = batch_size, importance = False, cuda = False)
    dataset_trn.on_epoch_end()
    dataset_vld.on_epoch_end()

    #########################################################################

    model = Model(time_steps = time_steps)
    model = model.cpu() if not CUDA else model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-6)

    OUTPUT_TIMESTAMP = str(datetime.now()).replace(" ", "_").split(":")[0]
    OUTPUT_BASE = join(BASE, "DELWAVEv1.0")
    if not exists(OUTPUT_BASE):
        mkdir(OUTPUT_BASE)

    #########################################################################

    best_train_loss = np.float32("inf")
    best_valid_loss  = np.float32("inf")

    train_loss_buffer = torch.zeros((epochs), dtype = torch.float32)
    valid_loss_buffer  = torch.zeros((epochs), dtype = torch.float32)

    for e in range(epochs):
 
        train_loss = 0.0
        valid_loss  = 0.0

        model.train()
        for i in range(len(dataset_trn)):

            x, y = dataset_trn.__getitem__(i)

            loss = model.loss(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            dataset_trn.importance = not dataset_trn.importance

        model.eval()
        with torch.no_grad():
            for i in range(len(dataset_vld)):

                x, y = dataset_vld.__getitem__(i)

                loss = model.loss(x, y)
                valid_loss += loss.item()

        dataset_trn.on_epoch_end()
        dataset_vld.on_epoch_end()

        if train_loss < best_train_loss:
            best_train_loss = train_loss


        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            x, _ = dataset_trn.__getitem__(0)

            tm = torch.jit.trace(model, x)
            torch.jit.save(tm, join(OUTPUT_BASE, "DELWAVE"))   

        train_loss_buffer[e] = train_loss/len(dataset_trn)
        valid_loss_buffer[e] = valid_loss/len(dataset_vld)

        print(f"Epoch: {e + 1}|\n--------------------\nTrain loss: {train_loss/len(dataset_trn)}\nTest loss: {valid_loss/len(dataset_vld)}")

        #scheduler.step(valid_loss)

        np.savetxt(join(OUTPUT_BASE, "Training_loss"), train_loss_buffer[0:e + 1].numpy())
        np.savetxt(join(OUTPUT_BASE, "Validation_loss"),  valid_loss_buffer[0:e + 1].numpy())

        #########################################################################
        # Early stopping

        e_cntr= e + 1
        if e_cntr > tolerance:

            min_val_loss = valid_loss_buffer[0:e_cntr].min()

            print(f"Early stopping check: ({min_val_loss} -> {valid_loss_buffer[e_cntr - tolerance:e_cntr]})")

            if not (min_val_loss in valid_loss_buffer[e_cntr - tolerance:e_cntr]):

                print(f"Early stopping at epoch {e_cntr}")
                break

        print()

main()
