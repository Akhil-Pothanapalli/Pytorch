# Image classifier

from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

# prepare the data
train = datasets.MNIST(
    root='data',
    train = True,
    download = True,
    transform = ToTensor()
)

dataset = DataLoader(train, 32)

# Define the Model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10),
        )

    def forward(self, x):
        return self.model(x)
    
# Instance the model
clf = ImageClassifier().to('cuda')
learning_rate = 1e-03
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(clf.parameters(), lr = learning_rate)

num_epochs = 10

# Driver code
if __name__ == "__main__":

  for epoch in range(num_epochs):
      for batch in dataset:
          X, y = batch
          
          X, y = X.to('cuda'), y.to('cuda')
          yhat = clf(X)

          loss = loss_fn(yhat, y)

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          print(f'epoch {epoch} loss is {loss.item()}')

# save the model
  with open('model.pt', 'wb') as f:
      save(clf.state_dict(), f)

#load the model with saved parameters
  with open('model.pt', 'rb') as f:
      clf.load_state_dict(load(f))

# Testing the data
  img = Image.open('image_path')
  img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

  print(torch.argmax(clf(img)))
