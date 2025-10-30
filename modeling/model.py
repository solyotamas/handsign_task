from turtle import forward
import torch
import torch.nn as nn

class SignLanguageCNN_pooling(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classes = num_classes

        # assuming 60 frame sequences
        # (32, 60, 1629)
        # need (32, 1629, 60)
        

        self.conv1 = nn.Conv1d(in_channels = 1629, out_channels = 64, kernel_size = 3, padding = 1)
        # -> x out channels
        # each kernel is (1629, 3) -> gives ( ,60) for 1 out_channel with padding = 1, without ( ,58)
        # x diff weighted kernels -> (x, 60)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size = 2) # 60 sequence length -> 30
        #(x,30)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
        # (x, 30) channels
        self.bn2   = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size = 2) # 30 sequence length -> 15, but channels up to capture deeper patterns tradeoff
        # (x,15)
        self.dropout2 = nn.Dropout(0.2)

        
        self.conv3 = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
        self.bn3   = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size = 2)
        #self.dropout3 = nn.Dropout(0.1)
        
        
        self.gap = nn.AdaptiveAvgPool1d(1)    # (32, x, 15)->(32, x, 1)
        self.fc = nn.Linear(128, self.classes)  # needs (32, x)  
    
    def forward(self, x):
        x = x.permute(0, 2, 1) 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



class SignLanguageSimple_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1629,     
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            #dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, 30, 1629)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = x = h_n[-1]  #x = torch.mean(lstm_out, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x