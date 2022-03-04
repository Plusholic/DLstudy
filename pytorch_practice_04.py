import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('c:/Users/bm990/Desktop/백업/Python_Code/DL/Pytorch_sample/data/reg.csv', index_col=[0])

X = df.drop('Price', axis=1).to_numpy()
Y = df['Price'].to_numpy().reshape((-1,1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

class TensorData(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
trainsets = TensorData(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)
testsets = TensorData(X_test, Y_test)
testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 50) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(50, 30) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(30, 1) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.5) # 연산이 될 때마다 50%의 비율로 랜덤하게 노드를 없앤다.
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


 
model = Regressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

loss_ = []
n = len(trainloader)

for epoch in range(400):
    running_loss = 0.0
    for data in trainloader:
        inputs, values = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, values)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    loss_.append(running_loss/n)
    
    
plt.plot(loss_)
plt.title("Training_loss")
plt.xlabel("epoch")
plt.show()

def evaluation(dataloader):
    
    predictions = torch.tensor([], dtype=torch.float)
    actual = torch.tensor([], dtype=torch.float)
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            inputs, values = data
            outputs = model(inputs)
            # print(outputs.shape)
            # print(predictions.shape)
            predictions = torch.cat((predictions, outputs), 0)
            actual = torch.cat((actual, values), 0)
            # print(actual.shape)
            
    predictions = predictions.numpy()
    # print(predictions.shape)
    actual = actual.numpy()
    # print(actual.shape)
    rmse = np.sqrt(mean_squared_error(predictions, actual))

    return rmse

train_rmse = evaluation(trainloader)
test_rmse = evaluation(testloader)
print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)
