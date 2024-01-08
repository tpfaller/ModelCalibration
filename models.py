import torch
from torch import nn
import torch.nn.functional as F

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
        

class CNNNet(nn.Module):
    def __init__(self) -> None:
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.sigmoid(x)
    
    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x).numpy()
