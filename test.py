from os import times_result
from os.path import join

from random import random
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from scipy.signal import medfilt

from datetime import datetime
import sys

import matplotlib
font = {"weight" : "normal", "size" : 14}
matplotlib.rc("font", **font)

import matplotlib.pyplot as plt

#########################################################################

#########################################################################

from dataset import Databank, Dataset
from spatial import spatial_map

#########################################################################

def main():
   
    CUDA = torch.cuda.is_available()

    #########################################################################

    if len(sys.argv) != 4:
        print("python3 test.py <dataset name> <model folder name> <base folder path>")
        exit()

    dataset_name = sys.argv[1]
    BASE = sys.argv[3]
    MODEL_NAME = join(BASE, sys.argv[2], "Model")

    dmap = {"AA": 0,
            "MB": 1,
            "GD": 2,
            "OB": 3,
            "OB2": 4,
            "OB3": 5}


    time_steps = 11
    batch_size = 16

    print("Reading dataset...")

    BASE_TR = join(BASE, "data")

    X = np.load(join(BASE_TR, "test_wind_field.npy"))
    Y = np.load(join(BASE_TR, "test_waves.npy"))
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
    databank = Databank(X, Y, S, time_steps = time_steps, normalize = np.load(join(BASE_TR, "normalization.npy")), station_indices = station_indices, cuda = False)
    indices_to_eval = databank.indices

    dataset = Dataset(databank, indices_to_eval, batch_size = batch_size, importance = False, randomize = False, cuda = CUDA)  
    batch_size = batch_size*Y.shape[0]

    #########################################################################

    model = torch.jit.load(MODEL_NAME)
    model.eval()

    model = model.cpu() if not CUDA else model

    #########################################################################

    P = np.zeros((indices_to_eval.size*Y.shape[0], Y.shape[2]), dtype = np.float32)
    loss = 0.0

    with torch.no_grad():
        for i in range(dataset.__len__()):
            
            x, y = dataset.__getitem__(i)
            p = model(x)

            loss += torch.sqrt(torch.pow(p - y, 2).mean())
            
            b0 = i*batch_size
            b1 = np.clip((i + 1)*batch_size, a_min = 0, a_max = P.shape[0])

            print(f"{i}: [{b0}|{b1}]{p.shape}/{P.shape[0]}")
            
            P[b0:b1, :] = p.detach().cpu().numpy()

            print(f"Loss {i + 1}: {loss/(i + 1)}")

    print(f"RME: {loss/len(dataset)}")

    Y = databank.Y.detach().cpu().numpy()
    Y = Y[time_steps - 1:, 0]

    #########################################################################
    
    # New norm_new.npy
    
    n = np.load(join(BASE_TR, "normalization.npy"))

    h_mean = n[0:6]
    p_mean = n[6:12]
    d_mean = n[12:18]

    h_std = n[18:24]
    p_std = n[24:30]
    d_std = n[30:36]

    x_mean = n[36]
    x_std  = n[37]
    
    idx = station_indices[-1]

    height_p = np.exp(np.clip(P[:, 0]*h_std[idx] + h_mean[idx], a_min = 0.0, a_max = None)) - 1.0
    height_t = np.exp(np.clip(Y[:, 0]*h_std[idx] + h_mean[idx], a_min = 0.0, a_max = None)) - 1.0
    height_p /= 100.0
    height_t /= 100.0
    height_min = np.array([height_p.min(), height_t.min()]).min()
    height_max = np.array([height_p.max(), height_t.max()]).max()

    period_p = np.clip(P[:, 1]*p_std[idx] + p_mean[idx], a_min = 0.0, a_max = None)/100.0
    period_t = (Y[:, 1]*p_std[idx] + p_mean[idx])/100.0
    period_min = np.array([period_p.min(), period_t.min()]).min()
    period_max = np.array([period_p.max(), period_t.max()]).max()

    direction_p = P[:, 2]*d_std[idx] + d_mean[idx]
    direction_p[direction_p < 0.0]   = 360.0 + direction_p[direction_p < 0.0]
    direction_p[direction_p > 360.0] = direction_p[direction_p > 360.0] - 360.0  
    direction_t = Y[:, 2]*d_std[idx] + d_mean[idx]
    direction_min = np.array([direction_p.min(), direction_t.min()]).min()
    direction_max = np.array([direction_p.max(), direction_t.max()]).max()

    def rmse(ix, iy):
        return np.sqrt(np.power(ix - iy, 2).mean())

    def mae(ix, iy):
        return np.abs(ix - iy).mean()

    print(f"Height RMSE: {rmse(height_p, height_t)} MAE: {mae(height_t, height_p)}")
    print(f"Period RMSE: {rmse(period_p, period_t)} MAE: {mae(period_t, period_p)}")
    print(f"Direction RMSE: {rmse(direction_p, direction_t)} MAE: {mae(direction_t, direction_p)}")

    ################################################################################################

    np.save(f"{dataset_name}_direction_y", direction_t)
    np.save(f"{dataset_name}_direction_p", direction_p)

    np.save(f"{dataset_name}_period_y", period_t)
    np.save(f"{dataset_name}_period_p", period_p)

    np.save(f"{dataset_name}_height_y", height_t)
    np.save(f"{dataset_name}_height_p", height_p)

    #########################################################################

    DPI = 110

    fig0, axs0 = plt.subplots(1, 3, figsize = (2560/DPI, 1440/DPI), dpi = DPI)
    fig1, axs1 = plt.subplots(1, 3, figsize = (2560/DPI, 1440/DPI), dpi = DPI)

    _plot_hist(axs0[0], period_p, period_t, period_min, period_max, 100, "Period [seconds]", "MAE [seconds]", "DELWAVE 2070-2100", "SWAN 2070-2100", f"{dataset_name}: Mean Wave Period MAE")
    _plot_hist(axs0[1], height_p, height_t, height_min, height_max, 100, "Height [meters]", "MAE [meters]", "DELWAVE 2070-2100", "SWAN 2070-2100", f"{dataset_name}: Significant Wave Height MAE")
    _plot_hist(axs0[2], direction_p, direction_t, direction_min, direction_max, 80, "Direction [degrees]", "MAE [degrees]", "DELWAVE 2070-2100", "SWAN 2070-2100", f"{dataset_name}: Mean Wave Direction MAE")

    _plot_scatter(axs1[0], period_p, period_t, "SWAN 2070-2100", "DELWAVE 2070-2100", f"{dataset_name}: Mean Wave Period")
    _plot_scatter(axs1[1], height_p, height_t, "SWAN 2070-2100", "DELWAVE 2070-2100", f"{dataset_name}: Significant Wave Height")
    _plot_scatter(axs1[2], direction_p, direction_t, "SWAN 2070-2100", "DELWAVE 2070-2100", f"{dataset_name}: Mean Wave Direction")


    fig0.savefig(f"{dataset_name}_hist.pdf",    bbox_inches = "tight", format = "pdf")
    fig1.savefig(f"{dataset_name}_scatter.png", bbox_inches = "tight", format = "png")

def _plot_scatter(ax, y0, y1, x_label, y_label, title):

        #color =  (72.0/255.0, 205.0/255.0, 211.0/255.0)
        color = (255.0/255.0, 161.0/255.0, 122.0/255.0)
        #color = (143.0/255.0, 188.0/255.0, 143.0/255.0)

        min_y = np.min([np.min(y0), np.min(y1)])
        max_y = np.max([np.max(y0), np.max(y1)])

        ax.scatter(y0, y1, alpha = 0.02, color = color)
        ax.plot([min_y, max_y], [min_y, max_y], ls = "--", color = "red", linewidth = 3)

        #ax.set_aspect(1)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        ax.set_title(title)

        ax.set_aspect("equal", adjustable = "box")
        ax.grid(True)

def _plot_hist(ax, y0, y1, range_min, range_max, n_bins, xlabel, ylabel, label0, label1, title):

    print(f"{y0.shape} {y1.shape}")

    # y0: predicted
    # y1: groundtruth

    mae = np.abs(y0 - y1)

    step = (range_max - range_min)/n_bins
    bins = np.arange(range_min, range_max, step)

    counts0, b_edge0 = np.histogram(y0, bins = bins)
    indices = np.digitize(y1, bins)
    counts1, b_edge1 = np.histogram(y1, bins = bins)

    mm = np.array([counts0.max(), counts1.max()]).max()

    counts0 = counts0/mm
    counts1 = counts1/mm

    mae_weights = np.zeros(b_edge0.size, dtype = np.float32)

    for i in np.unique(indices):
        mae_weights[i - 1] = mae[indices == i].mean()

    mae_weights = medfilt(mae_weights, kernel_size = 3)

    counts0 *= range_max*1.1#mae_weights.max()*1.5
    counts1 *= range_max*1.1#mae_weights.max()*1.5

    c_max = np.maximum(counts0.max(), counts1.max())
    c_min = np.minimum(counts0.min(), counts1.min())

    ax.hist(b_edge0[:-1], bins = bins, weights = counts0, label = label0, alpha = 0.8, color = (72.0/255.0, 205.0/255.0, 211.0/255.0))
    ax.hist(b_edge0[:-1], bins = bins, weights = counts1, label = label1, alpha = 0.7, color = (255.0/255.0, 161.0/255.0, 122.0/255.0))
    ax.plot(bins, mae_weights, color = "lightskyblue", label = "MAE DELWAVE", linewidth = 3)

    ax.set(xlabel = xlabel, ylabel = ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.grid(True)


main()

