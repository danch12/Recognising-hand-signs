import cv2
import os
import numpy as np
video_file_path="/Users/danielchow/Downloads/videos_for_nueral_net/mp4s/1_finger_stationary.mp4"
cam = cv2.VideoCapture(video_file_path) 
folder_name='data'+'_'+video_file_path.split('/')[-1].split('_')[0]+'_'+video_file_path.split('/')[-1].split('_')[-1].split('.')[0]+'gaussian'
fgbg= cv2.createBackgroundSubtractorMOG2()

try: 
      
    # creating a folder named data 
	if not os.path.exists(folder_name): 
		os.makedirs(folder_name) 
  
    # if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 
  
    # frame 
currentframe = 1
  
while(True): 
      
    # reading from frame 
	ret,frame = cam.read() 
  	
	if ret: 
    	# if video is still left continue creating images
		name = './'+folder_name+'/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name)
		cropped = frame[100:1100,50:500]
		height , width , layers = cropped.shape
		new_h=int(height/2)
		new_w=int(width/2)
		resize = cv2.resize(cropped, (new_w, new_h))
		fgmask=fgbg.apply(resize)
		median=cv2.medianBlur(fgmask,5)
		ret,thresh1 = cv2.threshold(median,20,1000,cv2.THRESH_BINARY)
		

    	# writing the extracted images 
		cv2.imwrite(name, thresh1) 
  
    	#stop duplicate images
		currentframe += 1
	else: 
		break
  
    #Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()