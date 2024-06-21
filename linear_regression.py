# import dependencies
import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data
X_numpy, y_numpy = datasets.make_regression(n_samples= 100, n_features =1, noise= 10, random_state=42)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)


n_samples, n_featues = X.shape
input_size = n_features
output_size = 1

# Define the model
model = nn.Linear(input_size, output_size)

# Hyper parameters
learning_rate = 1e-03
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 5000

# model optimization
for epoch in range(num_epochs):

    yhat = model(X)
    loss = criterion(yhat, y)

    # backpropagation and gradient descent
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1)%500 == 0:
        print(f'epoch {epoch+1} loss is {loss.item()}')

# predicting using the optimized model
predicted = model(X).detach().numpy()

# plotting the predicted line against the datapoints
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
