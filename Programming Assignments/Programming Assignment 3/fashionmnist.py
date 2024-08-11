import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''

'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64

'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

'''
PART 3:
Design a neural network in PyTorch. Architecture is up to you, but please ensure that the model achieves at least 80% accuracy.
Do not directly import or copy any existing models from other sources, spend time tweaking things.
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


net = Net()

'''
PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

'''
PART 5:
Train your model!
'''

num_epochs = 10
training_losses = []

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    training_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}, Training loss: {avg_loss:.4f}")

print('Finished Training')

torch.save(net.state_dict(), 'fmnist.pth')  # Saves model file (upload with submission)

'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''

correct = 0
total = 0
incorrect_image, incorrect_pred, incorrect_true = None, None, None
correct_image, correct_pred, correct_true = None, None, None
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                incorrect_image = images[i]
                incorrect_pred = classes[predicted[i]]
                incorrect_true = classes[labels[i]]
                break

        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                correct_image = images[i]
                correct_pred = classes[predicted[i]]
                correct_true = classes[labels[i]]
                break

        if incorrect_image is not None and correct_image is not None:
            break

print('Accuracy: ', correct / total)

'''
PART 7:
Check the written portion. You need to generate some plots. 
'''

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()


def imshow(img, predicted_label, true_label, filename):
    img = img / 2 + 0.5  # unnormalize the image
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.title(f'Predicted: {predicted_label}, True: {true_label}')
    plt.savefig(filename)
    plt.close()


imshow(incorrect_image, incorrect_pred, incorrect_true, 'incorrectly_classified.png')
imshow(correct_image, correct_pred, correct_true, 'correctly_classified.png')
