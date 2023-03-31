import torch

import numpy as np

def to_gpu(t):
    return t.cuda() if torch.cuda.is_available() else t

############################################################################

class Dataset:

    def __init__(self, databank, indices, batch_size = 32, importance = False, randomize = True, cuda = False):
        
        self.indices = indices
        self.batch_size = batch_size
        self.randomize = randomize
        self.cuda = cuda

        self.size = int(np.ceil(indices.size/batch_size))
        self.db = databank

        ### Importance sampling ###

        self.importance = importance
        self.import_nvrs = self.db.Y.shape[-1]
        self.import_nlcs = self.db.Y.shape[1]

        print(f"{self.import_nvrs} vars for {self.import_nlcs} locations")
        
        self.import_data = []

        if importance:
            for j in range(self.import_nlcs):

                per_lc_import = []

                for i in range(self.import_nvrs):

                    nbins = 100.0

                    y = databank.Y[indices][:, j, i].detach().cpu().numpy()

                    y_min = y.min()
                    y_max = y.max()
                    step = (y_max - y_min)/nbins

                    bins_raw = np.arange(y_min, y_max, step = step)

                    count, bins = np.histogram(y, bins = bins_raw)
                    idx = np.digitize(y, bins)

                    per_lc_import.append([idx, np.unique(idx)])

                self.import_data.append(per_lc_import)

            print(f"import_data: {len(self.import_data)}")

    def __getitem__(self, idx):

        if not self.importance:

            i0 = idx*self.batch_size
            i1 = np.clip((idx + 1)*self.batch_size, a_min = 0, a_max = self.indices.size)

            idx = self.indices[i0:i1]

            X = []
            dY = self.db.Y[idx]
            for i in range(self.db.time_steps):
                X.append(self.db.X[idx - i])
            dX = torch.stack(X, dim = 1)

            ii = self.db.I.expand(dX.shape[0], -1, -1, -1, -1)
            s = self.db.S.expand(dX.shape[0], -1, -1, -1, -1)

            X = []
            Y = []
            for i in range(self.db.ndatasets):

                Y.append(dY[:, i])
                ts = torch.unsqueeze(s[:, :, i], dim = 2)
                X.append(torch.cat([dX, ts, ii], dim = 2))

            dY = torch.cat(Y, dim = 0)
            dX = torch.cat(X, dim = 0)

        else:

            var_idx = np.random.randint(0, self.import_nvrs)

            spat = self.db.I.expand(self.batch_size, -1, -1, -1, -1)
            loca = self.db.S.expand(self.batch_size, -1, -1, -1, -1)

            combX = []
            combY = []

            for j in range(self.import_nlcs):

                import_data = self.import_data[j][var_idx]
                s = np.random.choice(import_data[1], size = self.batch_size)
                
                X = []
                Y = []
                for i in s:
                    idx_lcs = np.random.choice(self.indices[import_data[0] == i])
                    Y.append(self.db.Y[idx_lcs, j])

                    tX = []
                    for k in range(self.db.time_steps):
                        tX.append(self.db.X[idx_lcs - k])

                    X.append(torch.stack(tX, dim = 0))

                X = torch.stack(X, dim = 0)
                Y = torch.stack(Y, dim = 0)

                ts = torch.unsqueeze(loca[:, :, j], dim = 2)
                X = torch.cat([X, ts, spat], dim = 2)

                combX.append(X)
                combY.append(Y)

            dY = torch.cat(combY, dim = 0)
            dX = torch.cat(combX, dim = 0)
     
        #################################################

        if self.randomize:
            
            p = np.random.permutation(dY.shape[0])
            dX = dX[p]
            dY = dY[p]

        return (dX.cuda(), dY.cuda()) if self.cuda else (dX, dY)
        
    def __len__(self):
        return self.size

    def on_epoch_end(self):
        
        p = np.random.permutation(self.indices.size)
        self.indices = self.indices[p]

        if self.importance:
            for j in range(self.import_nlcs):
                for i in range(self.import_nvrs):
                    self.import_data[j][i][0] = self.import_data[j][i][0][p]
    
class Databank:

    def __init__(self, X, Y, S, time_steps = 11, normalize = None, station_indices = None, cuda = True):

        self.indices = np.arange(time_steps - 1, X.shape[0])

        self.X = torch.tensor(X, dtype = torch.float32, requires_grad = False)
        self.Y = torch.tensor(Y, dtype = torch.float32, requires_grad = False)
        self.S = torch.tensor(S, dtype = torch.float32, requires_grad = False)
        self.I = torch.tensor(np.arange(X.shape[-1]*X.shape[-2]), dtype = torch.float32, requires_grad = False)/(X.shape[-1]*X.shape[-2])

        #### Normalize data ####

        if normalize is not None:
            
                       
            # New norm_new.npy
            h_mean = normalize[0:6]
            p_mean = normalize[6:12]
            d_mean = normalize[12:18]

            h_std = normalize[18:24]
            p_std = normalize[24:30]
            d_std = normalize[30:36]

            x_mean = normalize[36]
            x_std  = normalize[37]

            self.X = (self.X - x_mean)/x_std

            for i in range(self.Y.shape[0]):
                
                idx = station_indices[i]

                self.Y[i, :, 0] = (torch.log(self.Y[i, :, 0] + 1.0) - h_mean[idx])/h_std[idx]
                self.Y[i, :, 1] = (self.Y[i, :, 1] - p_mean[idx])/p_std[idx]
                self.Y[i, :, 2] = (self.Y[i, :, 2] - d_mean[idx])/d_std[idx]

                splt = int(0.8*self.Y.shape[1])

                print(f"Station {i} height: {self.Y[i, :splt, 0].mean()} {self.Y[i, :, 0].std()}")
                print(f"Station {i} period: {self.Y[i, :splt, 1].mean()} {self.Y[i, :, 1].std()}")
                print(f"Station {i} direct: {self.Y[i, :splt, 2].mean()} {self.Y[i, :, 2].std()}\n")

        ########################

        self.ndatasets = self.Y.shape[0]
        self.time_steps = time_steps

        if cuda:

            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
            self.S = self.S.cuda()
            self.I = self.I.cuda()

        self.S = torch.reshape(self.S, (1, 1) + S.shape).expand(-1, time_steps, -1, -1, -1)

        self.Y = torch.swapaxes(self.Y, 0, 1)
        self.I = torch.reshape(self.I, (1, 1, 1, self.X.shape[-2], self.X.shape[-1])).expand(-1, time_steps, -1, -1, -1)


