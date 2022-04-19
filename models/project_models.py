import torch.nn as nn


class FC_500_150(nn.Module):
    """
    Fully connected linear model with two hidden layers, based on the model
    used in the DeepFool paper
    [https://arxiv.org/pdf/1511.04599.pdf]
    """
    
    def __init__(self):
        super(FC_500_150, self).__init__()
        
        self.activation = nn.ReLU()
        
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 150)
        self.fc3 = nn.Linear(150, 10)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x


class LeNet_CIFAR(nn.Module):
    """
    LeNet-5 model for the CIFAR-10 dataset, using three convolutional
    layers, based on the model used in the DeepFool paper
    [https://arxiv.org/pdf/1511.04599.pdf]
    """
    
    def __init__(self):
        super(LeNet_CIFAR, self).__init__()
        
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=5, stride=1, padding=2)
        
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=84)
        self.fc4 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return x


class LeNet_MNIST(nn.Module):
    """
    LeNet-5 model for the MNIST dataset
    """
    
    def __init__(self):
        super(LeNet_MNIST, self).__init__()
        
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x


class Net(nn.Module):
    """
    Network-In-Network model implemented using PyTorch
    [https://github.com/jiecaoyu/pytorch-nin-cifar10]
    """
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
            
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0))
    
    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        
        return x
