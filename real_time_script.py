import cv2
import os
import numpy as np

import pandas as pd
import imutils
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from tensorflow.keras.models import load_model


class_dict={0:'1_finger',
           1:'ok_sign',
           2:'peace_sign'}

def preprocess(image):
	image = np.stack((image,) * 3, axis=-1)
	resized_image = tf.image.resize(image, [224, 224])
	final_image = keras.applications.xception.preprocess_input(resized_image)
	return final_image




model=load_model("/Users/danielchow/Downloads/videos_for_nueral_net/models/best_home_model.h5")


prob=None
cl=None


cam = cv2.VideoCapture(0) 
fgbg= cv2.createBackgroundSubtractorMOG2()


  
    # frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
	ret,frame = cam.read() 
  	
	if ret: 
    	# if video is still left continue creating images

    	
    	# resize the frame
		

    	# clone the frame
		clone = frame.copy()
		name = './'+folder_name+'/frame' + str(currentframe) + '.jpg'
		#print ('Creating...' + name)
		cropped = frame[290:1100,200:1000]
		height , width , layers = cropped.shape
		new_h=int(height/2)
		new_w=int(width/2)
		resize = cv2.resize(cropped, (new_w, new_h))
		fgmask=fgbg.apply(resize)
		median=cv2.medianBlur(fgmask,5)
		ret,thresh1 = cv2.threshold(median,20,1000,cv2.THRESH_BINARY)
		print(thresh1.shape)

		


		if currentframe % 5 ==0:
			
			img=preprocess(thresh1)
			img=img/255
			im2 = np.expand_dims(img, axis=0)
			print(im2.shape)
			prob=max(model.predict(im2)[0])
			prob=round(prob,4)
			cl=class_dict[np.argmax(model.predict(im2))]
			font                   = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (100,200)
			fontScale              = 1
			fontColor              = (0,255,255)
			lineType               = 2



		cv2.putText(frame,
		f'{prob}-{cl}', 
		bottomLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)

    	# writing the extracted images 
		cv2.imshow('segment', thresh1) 
		cv2.imshow('real',frame)
    	#stop duplicate images
		currentframe += 1

		 # observe the keypress by the user
		keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
		if keypress == ord("q"):
			break 
	else: 
		break
  
    #Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()