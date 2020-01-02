# Recognising hand signals

Using a CNN to recognise various hand signals


## Table of Contents
* [Technologies Used]
* [Packages Used]
* [Methods Used]
* [Introduction]
* [Forming the Dataset]



## Technologies Used
* Python 3.0
* Jupyter Notebook
* Sublime Text
* Google Collab

## Packages Used
* Keras
* Tensorflow
* PIL
* CV2
* Matplotlib
* Numpy
* os
* Pandas
* PIL
* time


## Methods Used
* Image processing
* Image classification


## Introduction

The inspiration for this project came from watching a TV show where one character creates an AI that learns to predict violent acts across the world before they happen. This goal may be slightly too optimistic for me currently (who knows what the future may hold) but one of the first things the AI learns was how to recognise its creator. I thought I could do something similar but instead of recognising its creator, maybe I could create a AI (CNN) to recognise my hands in real time doing various signals. This is what I have set out to achieve.


## Forming the Dataset

[Link to Notebook section]

To train the convoluted nerual network I needed a Dataset of images. I saw that there were some quite good datasets around the web, like this [one](https://www.kaggle.com/gti-upm/leapgestrecog) but finally decided it would be a good way of improving my skills if I created my own dataset, especially as I had never used the cv2 library before. To create my own dataset I videoed myself doing three different hand signals into my webcam and then used the frames as the dataset. Additionally I was worried that the background of the video would have a large impact on the quality of the model predictions. Therefore I used a MOG Background subtractor to takeaway the background so that in theory only the hand was visible. Following this I used median blur and thresholding to get rid some of the irregularities within the frame. The only other preproccessing step was to crop the frame and resize it so that the model didn't have to process anything more than it had to. Below are some examples of the pictures in the dataset but if you would like to see them all they are [here].

#### 1 finger
![1_finger](https://raw.githubusercontent.com/danch12/Images_for_neural_hands/master/Data/Train/1_finger/frame138.jpg)

#### peace sign
![peace_sign](https://raw.githubusercontent.com/danch12/Images_for_neural_hands/master/Data/Train/peace_sign/frame159.jpg)

#### ok sign
![ok_sign](https://raw.githubusercontent.com/danch12/Images_for_neural_hands/master/Data/Train/ok_sign/frame1324.jpg)

## Training the Neural Network 

[Link to Notebook section]

The neural network that I used was very easy to implement once I did all of the above preproccessing steps, I used the fantastic Xception as a base model and then just put my own output layer on top as I only wanted it to predict the 3 classes defined above. Once I unfroze the base layers the model quickly achieved a validation accuracy of over 97% which I was happy with especially considering some of the images were not entirely clear due how the data was gathered. As the final model is too large for github, [here](https://drive.google.com/open?id=1qwIOlTQt92BF0mHoeLzuBZ1VYIvK-sf3) is a link to the google drive where it is saved.


## Predicting Real Time

[Link to Notebook section]

Finally to bring it all together I wanted the model to predict hand signals real time using my webcam. This required a couple changes to the original data gathering script but nothing too major, most notably have to process the images slightly furter so that they would be accepted by the model. An example video can be found [here] showcaing the model in action.


## Conclusion

Overall I am happy with how this mini project has turned out and having seen the capabilities of the OpenCV library I am very excited to expand this project further as even in my limited experience it has shown itself to be extremely powerful and interesting. In the future I would like to expand the project by getting the model to recognise my face and also want to include functionality to my hand signals. For example maybe when I do the ok sign it increases the volume on my laptop.




