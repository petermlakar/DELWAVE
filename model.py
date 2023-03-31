import torch
import torch.nn as nn
import numpy as np

from torch.nn.functional import pad


class StartBlock(nn.Module):

    def __init__(self, location_encoding):

        super().__init__()

        self.reduce = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.b0 = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size = 3, stride = 1, padding = (1, 1)),
                nn.SiLU())

        self.b1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 3, padding = (1, 1)),
                nn.SiLU())

    def forward(self, x):

        x = self.reduce(self.b0(x))
        x = x + self.b1(x)

        return x

class EndBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.b0 = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size = 1, stride = 1),
                nn.SiLU())

    def forward(self, x):
        return self.b0(x)

class EncoderBlock(nn.Module):

    def __init__(self):
    
        super().__init__()

        self.reduce = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.b0 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = (1, 1)),
                nn.SiLU())

        self.b1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = (1, 1)),
                nn.SiLU())

        self.b2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = (1, 1)),
                nn.SiLU())
        
        self.b3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = (1, 1)),
                nn.SiLU())

    def forward(self, x):
        
        x = self.reduce(self.b0(x) + x)
        x = self.reduce(self.b1(x) + x)
        x = self.reduce(self.b2(x) + x)
        x = self.b3(x) + x

        return x

class ReductionBlock(nn.Module):

    def __init__(self, time_steps):
    
        super().__init__()

        self.b0 = nn.Sequential(
                nn.Conv1d(time_steps, time_steps, kernel_size = 3, stride = 1, padding = 1),
                nn.SiLU(),

                nn.Conv1d(time_steps, 1, kernel_size = 1, stride = 1))

    def forward(self, x):
        return self.b0(x)

class RegressionBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.b0 = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.SiLU())

        self.b1 = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.SiLU())

        self.b2 = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(256, 256),
                nn.SiLU())

        self.b3 = nn.Linear(256, 3)

    def forward(self, x):
        
        x = x + self.b0(x) 
        x = x + self.b1(x)
        x = x + self.b2(x)

        return self.b3(x)

class Model(nn.Module):
    
    def __init__(self, time_steps, location_encoding = True):
 
        super(Model, self).__init__()

        self.location_encoding = location_encoding

        self.start = nn.ModuleList()
        self.end   = nn.ModuleList()

        self.encoder = EncoderBlock()

        for i in range(time_steps):
            self.start.append(StartBlock(location_encoding))
            self.end.append(EndBlock())

        self.reduce  = ReductionBlock(time_steps)
        self.regress = RegressionBlock()

    def forward(self, x):

        bs = x.shape[0]

        enc = []
        for idx, m in enumerate(self.start):

            p = self.end[idx](self.encoder(m(x[:, idx])))
            p = p.sum(dim = -1).sum(dim = -1)

            enc.append(p)

        xp = self.reduce(torch.stack(enc, dim = -2))
        return self.regress(xp.view((bs, 256)))

    def loss(self, x, y):
        
        p = self(x)
        l = torch.pow(p - y, 2)

        return torch.sqrt(torch.cat([torch.unsqueeze(l[:, 0]*2.0, dim = -1), l[:, 1:]], dim = -1).mean())

    def set(self, m):
       
        m_p = m.state_dict()

        for p_name, p in self.named_parameters():

            if p_name in m_p.keys():
                p.data = m_p[p_name]    
            else:
                print(f"{p_name} not in model...")

    def get_name(self):
        return "DELWAVEv1.0"


