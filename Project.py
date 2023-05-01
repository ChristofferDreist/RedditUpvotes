# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:26:10 2023

@author: sebastian
"""

## Important to set correct working directory and to add a folder called Data in order to collect relevant data

#%%
#Initial load
from RedDownloader import RedDownloader
import praw
import requests
import re
import os
import urllib.request
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from sklearn import metrics
import sys
from torchvision import datasets, transforms, utils
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score, train_test_split
from scipy.stats import uniform
import random
from datetime import datetime
import os

if not os.path.exists("./Data"):
    os.makedirs("./Data")

#%%
reddit = praw.Reddit(
    client_id = "qK4Xv6veQzln_8kyWjqbSw",
    client_secret = "bvzcM1BI3Lv3vWAj4UmWU2GNjs1VIw",
    username = "AllHailAI",
    password = "ChrisSebChris",
    user_agent = "Test", 
    check_for_async = False ## added because of following message "It appears that you are using PRAW in an asynchronous environment. It is strongly recommended to use Async PRAW: https://asyncpraw.readthedocs.io.See https://praw.readthedocs.io/en/latest/getting_started/multiple_instances.html#discord-bots-and-asynchronous-environments for more info."
)

subreddit = reddit.subreddit('EarthPorn')
top = subreddit.top()

#%%
## Iterate through top pictures in subreddit. Can't download pictures from deleted accounts. Those are skipped
n_pictures = 1000

submission_ids = []
upvote_ratio = []
upvote = []
channels = []
num_comments = []
timestamps = []

for submission in subreddit.top(limit = n_pictures):
    if submission.url.endswith('.jpg') or submission.url.endswith('.png'):

        try:
            urllib.request.urlretrieve(submission.url, "./Data/{filename}".format(filename = f"{submission.id}.{submission.url.split('.')[-1]}"))
            img = Image.open("./Data/{filename}.jpg".format(filename = submission.id))
            img = img.resize((224,224))

            img.save("./Data/{filename}.jpg".format(filename = submission.id))
            submission_ids.append(submission.id)
            upvote_ratio.append(submission.upvote_ratio)
            upvote.append(submission.score)
            channels.append(len(img.getbands()))
            num_comments.append(submission.num_comments)
            timestamps.append(pd.to_datetime(submission.created_utc, unit ="s"))
            

        except Exception as e:
            print(e)
            pass

#%% 
# check if all correct data is collected for each parameter/header 
print(len(submission_ids))
print(len(upvote_ratio))
print(len(upvote))
print(len(channels))
print(len(num_comments))
#print(timestamps)
#%% 
df = pd.DataFrame({'submission_id':submission_ids, "upvote_ratio":upvote_ratio, "upvote":upvote, "channels":channels, "num_comments":num_comments, "timestamps":timestamps})

print(df.head())

#%%
random.seed(420)

# Normalize to mean and std of ImageNet
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),  
    ]
)

#%%
#Define CNN

# Learning rate moved up in order to use earlier
learning_rate = 1e-4 

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

    
    #Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1)

    #Define pooling layers
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride = 2)


    #Define fully connected layers
        self.fc1 = nn.Linear(128*28*28,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,1)

    #Dropout some neurons to prevent overfitting.
        self.dropout = nn.Dropout(0.5)

    #Define activation functions
        self.relu = nn.ReLU()
        self.identity = nn.Identity() # final layer changed from sigmoid to identity (Regression)


#Apply convolutional layers with pooling in between
    def forward(self, x):
        x = self.max_pool(nn.functional.relu(self.conv1(x)))
        x = self.max_pool(nn.functional.relu(self.conv2(x)))
        x = self.max_pool(nn.functional.relu(self.conv3(x)))

#Flatten output
        x = x.view(-1, 128*28*28)

        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.identity(self.fc3(x))

        return x

model = Model()
device = torch.device('cpu')  # use cuda or cpu
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(model)

#%%
#Loading data

data_path = "./Data"
filenames = df.submission_id
upvote_ratio = df.upvote_ratio

train_ids, test_ids, train_upvoteratios, test_upvoteratios = train_test_split(filenames, upvote_ratio, test_size=0.2, random_state = 42)

train_data = [(os.path.join(data_path, train_id + '.jpg'), train_upvoteratio) for train_id, train_upvoteratio in zip(train_ids, train_upvoteratios)]
test_data = [(os.path.join(data_path, test_id + '.jpg'), test_upvoteratio) for test_id, test_upvoteratio in zip(test_ids, test_upvoteratios)]

print(train_data)

print(len(train_data))
print(len(test_data))

#%%


# fix error "NoneType" object has no attribute 'shape' line 206
class Loader(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        img = self.transform(Image.open(path))

        return img, label


train_dataset = Loader(train_data, transform=transform)
test_dataset = Loader(test_data, transform=transform)

print('Model parameters:')
for name, param in model.named_parameters():
    print(f'{name}: {param.shape}')
    print(f'Gradient: {param.grad.shape}')

#%%
# Train loop
# Error line 253
batch_size = 10
num_epochs = 10
learning_rate = 1e-4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def CNN(learning_rate=learning_rate, batch_size= batch_size,num_epochs= num_epochs):

    criterion = nn.MSELoss() #uses a regression loss function instead of a binary cross-entropy loss.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for i, (images, ratio) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, ratio.float())
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        #train_losses.append(loss.item())

        # Test the model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_fn(output, target).item() # name 'loss_fn' is not defined
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
                
        test_losses.append(test_loss)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, avg_train_loss, avg_test_loss))

    print("Finished training.")
        
    return model, train_losses, test_losses

model, train_losses, test_losses = CNN(learning_rate=1e-4, batch_size=10, num_epochs=10)

#%%
# Choose a random image and normalize it
image = Image.open("{}/{}".format(data_path, "5ceg1q.jpg"))

plt.imshow(image)
plt.show()

image_tensor = transform(image).unsqueeze(0)
img = image_tensor.clone().detach().requires_grad_(True)


# Set up the optimizer
optimizer = optim.Adam([img.requires_grad_()], lr=1e-2)

# Define the target output (near zero)
target = torch.tensor([[0]]).float()

# Run the optimization loop
tv_weight = 1e-6
for i in range(1000):
    optimizer.zero_grad()
    output = model(img)
    loss = criterion(output, target) + tv_weight * torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + tv_weight * torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    loss.backward()
    optimizer.step()
    
    # Clip the pixel values to stay within [0, 1] range
    img.data = torch.clamp(img.data, 0, 1)
    
    # Print the loss every 100 iterations
    if i % 100 == 0:
        print("Iteration {}: Loss={}".format(i, loss.item()))
        plt.imshow(img.squeeze().permute(1, 2, 0).detach().numpy())
        plt.show()