import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print("CUDA is not available. Training on CPU.")
else:
    print("CUDA is available. Training on GPU.")

#Data loading from cat or dog dir.
batch_size = 20
data_dir = "Cat_Dog_data"
transform = transforms.Compose([transforms.Resize(255),
                                transforms.RandomRotation(30),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
train_data = datasets.ImageFolder(data_dir + "/train", transform=transform)
test_data = datasets.ImageFolder(data_dir + "/test", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

class Net(nn.Module):#input image is 224*224
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #224*224*3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) #56*56*16
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) #14*14*64
        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 500)
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) #56*56*16
        x = self.pool1(F.relu(self.conv2(x))) #14*14*32
        x = self.pool2(F.relu(self.conv3(x))) #7*7*64
        x = x.view(-1, 64*7*7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
model = Net()
print(model)

if train_on_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
n_epochs = 20
valid_loss_mi = np.Inf
train_losses, test_losses = [], []
for epoch in range(1, n_epochs+1):
    train_loss = 0
    model.train()
    for images, labels in train_loader:
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() #why multiple batch size... 
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in test_loader:
                output = model(images)
                test_loss += criterion(output, labels)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        
        print("Epoch: {}/{}..".format(epoch, n_epochs),
              "Training Loss: {:.3f}..".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

plt.plot(train_losses, lable="Training loss")
plt.plot(test_losses, lable="Validation loss")
plt.legend(frameon=False)

