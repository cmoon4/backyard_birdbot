# bird_vision
## Introduction
This is a silly hobby project to use existing ML models to:
  1. Detect any birds sighted by a webcam
  2. Identify which species they belongs to
  3. Post images and descriptions of the detected birds to twitter ([@BackyardBirdbot](https://twitter.com/BackyardBirdbot))

This project was my first Python project, so my main goal was to learn more about Python through experience. The entire program is run through *bird_detect.py*.

## Methods
As stated, the aim of the project is to use existing ML models to first detect birds then classify what species it belongs to. We won't be training any new models here. For object detection, we use the SSD Openimages v4 model published as part of TensorFlow Object Detection API (https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1). For classifying bird species, we fortunately have a lightweight bird species classification model also by TensorFlow/Google (https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1). We'll use OpenCV to capture our image and feed it to the models. 

We first import some libraries and some simple helper functions (the rest can be seen in the actual file: *bird_detect.py*):

```python
# <codecell> Importing libraries
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tweepy
import config
import os

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# <codecell> Helper functions (Not all are used, this section could be cleaned up)

def im_box_crop(img,box):
    # a function to crop an image using the normalized coordinates indicated by box output from the object detection model.
    im_height, im_width = img.shape[0], img.shape[1]
    ymin  = box[0]
    xmin  = box[1]
    ymax  = box[2]
    xmax  = box[3]
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    a,b,c,d = int(left) , int(right) , int(top) ,int(bottom)
    img_crop = img[c:d,a:b]
    return img_crop
    
def an_or_a(string):
    # a function to determine if the bird name should be prefaced with "a" or "an". Inspired by MIT course 6.0001 material (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)
    an_letters="aefhilmnorsxAEFHILMNORSX"
    char=string[0]
    if char in an_letters:
        output="an"
    else:
        output="a"
    return output
.
.
.
```

We then need to load the actual models themselves, and initialize the Twitter API through tweepy so that we can post
```
# <codecell> Load models
print('Loading detection/classification models...')
# for the main image detection model
#module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

detector = hub.load(module_handle).signatures['default']

# for the secondary bird classification model
module_b_handle="https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
detector_b=hub.load(module_b_handle).signatures['default']

# we also need to load the labelmap that will correlate the output to the actual
# species name.
df_bird=pd.read_csv('aiy_birds_V1_labelmap_amended.csv')
print('Models loaded!')

# <codecell> setup twitter api

auth = tweepy.OAuthHandler(
        config.twitter_auth_keys['consumer_key'],
        config.twitter_auth_keys['consumer_secret']
        )
auth.set_access_token(
        config.twitter_auth_keys['access_token'],
        config.twitter_auth_keys['access_token_secret']
        )
api = tweepy.API(auth)
```

