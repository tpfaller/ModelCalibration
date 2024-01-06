import torch
from torch import nn

class LogRegression(nn.Module):
    def __init__(self, in_features):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=2) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

    def predict_proba(self, x):
        x = torch.Tensor(x).to("cpu")
        with torch.no_grad():
            return self.forward(x).numpy()
        