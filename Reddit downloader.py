
from RedDownloader import RedDownloader
import praw
import requests
import re
import os
import urllib.request
from PIL import Image


## Test for at se om det downloader
# RedDownloader.Download("https://www.reddit.com/r/EarthPorn/comments/7t0swm/so_glad_i_got_up_for_this_15f_sunrise_theres_only/", output = 'Test', destination ="./Data")


##Download images
#posts = RedDownloader.DownloadBySubreddit('EarthPorn', 10)
#authors = posts.GetPostAuthors()
#title = posts.GetPostTitles


#clientID
#  qK4Xv6veQzln_8kyWjqbSw
# bvzcM1BI3Lv3vWAj4UmWU2GNjs1VIw
<<<<<<< HEAD
# 
=======

>>>>>>> CNN
reddit = praw.Reddit(
    client_id = "qK4Xv6veQzln_8kyWjqbSw",
    client_secret = "bvzcM1BI3Lv3vWAj4UmWU2GNjs1VIw",
    username = "AllHailAI",
    password = "ChrisSebChris",
    user_agent = "Test"
)

subreddit = reddit.subreddit('EarthPorn')
top = subreddit.top()


## Iterate through top pictures in subreddit. Can't download pictures from deleted accounts. Those are skipped
<<<<<<< HEAD
n_pictures = 1000
=======
n_pictures = 100

submission_ids = []
upvote_ratio = []
score = []
>>>>>>> CNN

for submission in subreddit.top(limit = n_pictures):
    if submission.url.endswith('.jpg') or submission.url.endswith('.png'):
        try:
            urllib.request.urlretrieve(submission.url, "./Data/{filename}".format(filename = f"{submission.id}.{submission.url.split('.')[-1]}"))
            img = Image.open("./Data/{filename}.jpg".format(filename = submission.id))
            img = img.resize((224,224))

            img.save("./Data/{filename}.jpg".format(filename = submission.id))
<<<<<<< HEAD
=======

            submission_ids.append(submission_ids)
            upvote_ratio.append(submission.upvote_ratio)
            score.append(submission.score)


>>>>>>> CNN
        except:
            pass






#VAE 
#Varionational auto encoder