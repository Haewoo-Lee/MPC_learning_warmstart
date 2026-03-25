import numpy as np
import scipy.io as sio
import torch
from torch import nn


class MLPWarmStart(nn.Module):
    def __init__(self, in_dim=25, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# load normalization
norm = np.load("mlp_warmstart_norm.npz")
x_mean = norm["x_mean"].astype(np.float64)
x_std  = norm["x_std"].astype(np.float64)
y_mean = norm["y_mean"].astype(np.float64)
y_std  = norm["y_std"].astype(np.float64)

# build model and load weights
model = MLPWarmStart(in_dim=25, out_dim=10) #빈 NN 구성하고
state = torch.load("mlp_warmstart_model.pt", map_location="cpu") #저장된 NN의 layer와 weight bias를 불러와서
model.load_state_dict(state) #빈 NN dictonary에 붙이는 작업
model.eval() #가중치 추출전에 하는 것은 안전장치이자 습관... 코드가 꼬이지 않게 하기 위한 것.

# extract weights
W1 = model.net[0].weight.detach().cpu().numpy().astype(np.float64) #Sequential안에 넣은 부품들에 순서가 있는것. net[0] : nn.Linear(in_dim, 64), net[1] : nn.ReLU(), net[2] : nn.Linear(64, 64), ...
b1 = model.net[0].bias.detach().cpu().numpy().astype(np.float64)

W2 = model.net[2].weight.detach().cpu().numpy().astype(np.float64)
b2 = model.net[2].bias.detach().cpu().numpy().astype(np.float64)

W3 = model.net[4].weight.detach().cpu().numpy().astype(np.float64)
b3 = model.net[4].bias.detach().cpu().numpy().astype(np.float64)

sio.savemat(
    "mlp_warmstart_weights.mat",
    {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
)

print("saved: mlp_warmstart_weights.mat")