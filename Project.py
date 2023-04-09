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
n_pictures = 200

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
print(timestamps)
#%% 
df = pd.DataFrame({'submission_id':submission_ids, "upvote_ratio":upvote_ratio, "upvote":upvote, "channels":channels, "num_comments":num_comments, "timestamps":timestamps})

print(df.head())
