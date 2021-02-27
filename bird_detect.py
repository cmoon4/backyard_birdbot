# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:19:49 2021

@author: cmoon
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv

print('Loading detection/classification models...')
# for the main image detection model
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(module_handle).signatures['default']

# for the secondary bird classification model
module_b_handle="https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
detector_b=hub.load(module_handle).signatures['default']

print('Models loaded!')


# Wait a millisecond
key=cv.waitKey(1)
# Use the front webcam
webcam = cv.VideoCapture(1)

# minimum score for the model to register it as a bird
minThresh=0.2

while True:
    try:
        # Acquire the image from the webcam
        check, frame = webcam.read()
        #print("Image Acquired")
    
        # Convert the frame into a format tensorflow likes
        converted_img  = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis, ...]
        
        # Run the image through the model
        result = detector(converted_img)
        
        # create empty arrays 
        name_B=[]
        score_B=[]
        box_B=[]
        
        # Loop through the results and see if any are "Birds" 
        # and if there are, store them in to the empty arrays
        for name, score, box in zip(result['detection_class_entities'], result['detection_scores'], result['detection_boxes']):
            if name=='Bird':
                if score>=minThresh:
                    name_B.append(name)
                    score_B.append(score.numpy()*100)
                    box_B.append(box.numpy())
        
        # if any birds were found
        if len(name_B)>0:
            
            # if a single bird was found, print the results
            if len(name_B)==1:
                print('I have found a bird! Score =', np.round(score_B,2))
            
            # if multiple birds were found, print the results
            else:
                print('I have found',len(name_B),'birds! Scores =', np.round(score_B,2))
        
        # wait 0.1 seconds and loop again
        key=cv.waitKey(100)
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        print("Camera off.")
        print("Program ended.")
        break    
            
cv.destroyAllWindows()
webcam.release()        
