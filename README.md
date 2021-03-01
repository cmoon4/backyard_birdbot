# bird_vision
## Introduction
This is a silly hobby project to use existing ML models to:
  1. Detect any birds sighted by a webcam
  2. Identify which species they belongs to
  3. Post images and descriptions of the detected birds to twitter ([@BackyardBirdbot](https://twitter.com/BackyardBirdbot))

This project was my first Python project, so my main goal was to learn more about Python through experience. The entire program is run through *bird_detect.py*.

## Methods
As stated, the aim of the project is to use existing ML models to first detect birds then classify what species it belongs to. We won't be training any new models here. For object detection, we use the SSD Openimages v4 model published as part of TensorFlow Object Detection API (https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1). 

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
```
