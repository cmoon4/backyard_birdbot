# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:19:49 2021

@author: cmoon
"""
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
import time

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
# <codecell> Inputs
# minimum score for the model to register it as a bird
minThresh=0.30
# minimum score for the identification model
minIdentThresh=0.33
# tweet cooldown (minutes for the program to wait before tweeting again)
twt_cd=5

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

# Below are helper functions directly from or inspired by Tensorflow Object Detection API.
def display_image(image):
    fig = plt.figure(figsize=(20,15))
    plt.grid(False)
    plt.imshow(image)
  
def display_image_title(image,title):
    fig = plt.figure(figsize=(20,15))
    plt.grid(False)
    plt.imshow(image)
    plt.title(title)
    
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("C:\Windows\Fonts/Arial.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


# <codecell> Load models
print('Loading detection/classification models...')
# for the main image detection model
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
#module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

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

# <codecell> Image acquisition
# Wait a millisecond
key=cv.waitKey(1)
# Use the front webcam
webcam = cv.VideoCapture(1)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, 2560)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 1920)
counter=0
while True:
    try:
        # Acquire the image from the webcam
        check, frame = webcam.read()
        
        #get the webcam size
        height, width, channels = frame.shape
        scale=25
        #prepare the crop
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*height/100),int(scale*width/130)
        
        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY
        
        cropped = frame[minX:maxX, minY:maxY]
        #print("Image Acquired")
        frame = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
        #display_image(frame)
        # Convert the frame into a format tensorflow likes
        converted_img  = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis, ...]
        #channels = tf.unstack (converted_img, axis=-1)
        #converted_img    = tf.stack   ([channels[2], channels[1], channels[0]], axis=-1)
        # Run the image through the model
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()
        #print("Inference time: ",end_time-start_time)
        
        # create empty dict 
        result_bird={"names":[],"scores":[],"boxes":[]}
        # Loop through the results and see if any are "Birds" 
        # and if there are, store them in to the empty dictionary
        for name, score, box in zip(result['detection_class_entities'], result['detection_scores'], result['detection_boxes']):
            if name=='Bird':
                if score>=minThresh:
                    result_bird["names"].append(name)
                    result_bird["scores"].append(score)
                    result_bird["boxes"].append(box)
        
        # create empty lists that will contain the name and the score for bird species identification
        ident_l=[]
        score_l=[]
        # if any birds were found
        num_bird=np.size(result_bird["names"])
        #print(num_bird)
        #display_image(frame)
        # just for plotting 
        if False:
            display_image(frame)
        if False:
            result_plt={key:value.numpy() for key,value in result.items()}
            image_with_boxes = draw_boxes(frame,result_plt['detection_boxes'],\
                                  result_plt["detection_class_entities"],\
                                      result_plt["detection_scores"])
            display_image(image_with_boxes)
        if num_bird>0:
            # squish the dictionary to a more useful format
            result_bird={"names":tf.concat(axis=0,values=result_bird["names"]),\
                             "scores":tf.concat(axis=0,values=result_bird["scores"]),\
                             "boxes":tf.stack(result_bird["boxes"],axis=0.5)}    
            
            # indices that will be used for image cropping (essentially an array of zeros)    
            box_indices=tf.zeros(shape=(num_bird,),dtype=tf.int32)   
            
            # crop the image into the different boxes where birds were detected
            cropped_img=tf.image.crop_and_resize(converted_img,result_bird["boxes"],box_indices,[224,224])
           
            img_crop=[]
            # for each cropped box,
            for image_index in range(num_bird):
                # reshape the image into the input format the classification model wants
                input_img=tf.reshape(cropped_img[image_index],[1,224,224,3])
                # put the image into the classication model
                det_out=detector_b(input_img)
                # which ID # does the model think is most likely?
                out_idx=np.argmax(det_out["default"].numpy())
                # and how confident is the model?
                out_score=np.round(100*np.max(det_out["default"].numpy()),1)
                # if the score is greater than the minimum thershold:
                if out_score>=minIdentThresh*100:
                    # get the bird's common name
                    temp_df=df_bird[df_bird.id==out_idx]
                    out_string=temp_df["common_name"].values[0]
                    
                    if out_string!="background":
                        # recrop the image here
                        box_crop=result_bird["boxes"][image_index].numpy()
                        bird_crop_img=(im_box_crop(frame,box_crop))
                        # convert it back to cv2 format (BGR)
                        bird_crop_img = cv.cvtColor(bird_crop_img, cv.COLOR_RGB2BGR)
                        # save the recropped image for posting
                        bird_crop_img_filename="Cropped_Bird_{}.jpg".format(image_index)
                        cv.imwrite(bird_crop_img_filename,bird_crop_img)
    
                        # append the name and score to the empty lists
                        ident_l.append(out_string)
                        score_l.append(out_score)
                        # plotting stuff (to be removed)
                        # temp_str=out_string+" Score:"+str(out_score)
                        # display_image_title(np.squeeze(input_img.numpy()),temp_str)
            # if any birds were detected and successfully identified:
                
            if len(ident_l)>0:     
                # save the captured frame first
                bird_img_filename="captured_frame.jpg"
                frame_bgr=cv.cvtColor(frame,cv.COLOR_RGB2BGR)
                cv.imwrite(bird_img_filename,frame_bgr)
                
                # if a single bird was found
                if len(ident_l)==1:
                    str_1="I have found a bird! I think it's"
                    str_2=an_or_a(ident_l[0])
                    combined_str="{} {} {} ({}%)".format(str_1,str_2,*ident_l,str(*score_l))
                    print(combined_str)

                # if multiple birds were found
                else:
                    str_1="I have found"
                    str_2=str(len(ident_l))
                    str_3="birds! I think they are:"
                    bird_out_string=[]
                    for bird_index in range(len(ident_l)):
                        bird_species_str=ident_l[bird_index]
                        bird_score_str=str(score_l[bird_index])
                        out_string="{} ({}%)".format(bird_species_str,bird_score_str)
                        bird_out_string.append(out_string)
                    separator=", "
                    combined_str="{} {} {} {}".format(str_1,str_2,str_3,separator.join(bird_out_string))
                    print(combined_str)
                    
                img_upload_paths=[bird_img_filename]
                img_upload_paths.extend(["Cropped_Bird_{}.jpg".format(i) for i in range(len(ident_l))])
                img_ids=[api.media_upload(i).media_id_string for i in img_upload_paths]
                api.update_status(status=combined_str,media_ids=img_ids)
                
                for i in range(len(img_upload_paths)):
                    os.remove(img_upload_paths[i])
                
                print("Tweet posted! Waiting for 1 minute")
                key=cv.waitKey(twt_cd*60000)
            else:
                print("Bird detected but no species identification.")
		# record frame for future investigation
                fail_file_name="dud_{}.jpg".format(str(counter))
                cv.imwrite(fail_file_name,cv.cvtColor(frame,cv.COLOR_RGB2BGR))
                counter=counter+1

        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        print("Camera off.")
        print("Program ended.")
        break    
            
webcam.release()        
