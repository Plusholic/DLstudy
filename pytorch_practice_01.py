import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

x = torch.FloatTensor(range(5))
print(x)
x = torch.FloatTensor(range(5)).unsqueeze(1)
print(x)
print(x.shape)

y = 2*x + torch.rand(5,1)
print(y)

num_feature = x.shape[1]
print(num_feature)

w = torch.randn(num_feature, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 1e-3
optimizer = torch.optim.SGD([w, b], lr=learning_rate)

loss_stack = []

for epoch in range(1001):
    y_hat = torch.matmul(x, w) + b
    loss = torch.mean((y_hat-y)**2)
    loss.backward()
    optimizer.step()
    loss_stack.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}:{loss.item()}')
        
with torch.no_grad():
    y_hat = torch.matmul(x, w) + b
    
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(loss_stack)
plt.title("Loss")
plt.subplot(122)
plt.plot(x,y,'.b')
plt.plot(x,y_hat,'-r')
plt.legend(['ground trouth','prediction'])
plt.title("Prediction")
plt.show()