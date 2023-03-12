
from RedDownloader import RedDownloader
import praw
import requests
import re
import os
import urllib.request


## Test for at se om det downloader
# RedDownloader.Download("https://www.reddit.com/r/EarthPorn/comments/7t0swm/so_glad_i_got_up_for_this_15f_sunrise_theres_only/", output = 'Test', destination ="./Data")


##Download images
#posts = RedDownloader.DownloadBySubreddit('EarthPorn', 10)
#authors = posts.GetPostAuthors()
#title = posts.GetPostTitles


#clientID
#  qK4Xv6veQzln_8kyWjqbSw
# bvzcM1BI3Lv3vWAj4UmWU2GNjs1VIw
# 
reddit = praw.Reddit(
    client_id = "qK4Xv6veQzln_8kyWjqbSw",
    client_secret = "bvzcM1BI3Lv3vWAj4UmWU2GNjs1VIw",
    username = "AllHailAI",
    password = "ChrisSebChris",
    user_agent = "Test"
)

subreddit = reddit.subreddit('EarthPorn')
top = subreddit.top(limit = 10)


for submission in subreddit.top(limit = 10):
    if submission.url.endswith('.jpg') or submission.url.endswith('.png'):
        urllib.request.urlretrieve(submission.url, "./Data/{filename}".format(filename = submission.url.split('?')[0].split('/')[-1]))





