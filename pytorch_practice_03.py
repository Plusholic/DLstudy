import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.FloatTensor(range(5)).unsqueeze(1)
y = 2*x + torch.rand(5,1)


class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1,1, bias=True)
        
    def forward(self, x):
        y = self.fc(x)
        
        return y
    
model = LinearRegressor()
learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

loss_stack = []
for epoch in range(1001):
    
    optimizer.zero_grad()
    
    y_hat = model(x)
    loss = criterion(y_hat, y)
    
    loss.backward()
    optimizer.step()
    loss_stack.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'epoch : {epoch}, loss: {loss.item()}')
        
    
with torch.no_grad():
    y_hat = model(x)
    
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(loss_stack)
plt.title("Loss")
plt.subplot(122)
plt.plot(x,y,'.b')
plt.plot(x,y_hat,'r-')
plt.legend(['ground truth', 'prediction'])
plt.title("Prediction")
plt.show()