# Emotion Detection
Real-time face detection and emotion classification using fer2013 dataset, opencv/retinaface for face detection and xception and cnn models for emotion classification.

Emotion classification metrics
-  accuracy ~ 64%
-  top 2 accurasy ~ 81%
-  cross-entropy loss ~ 1.08

# Examples 

<table>
  <tr>
    <td align='center'>Happy</td>
     <td align='center' >Angry</td>
     <td align='center' >Sad</td>
  </tr>
  <tr>
    <td><img src="example_images/happy_pred.jpg" width=270 height=200></td>
    <td><img src="example_images/angry_pred.jpg" width=270 height=200></td>
    <td><img src="example_images/sad_pred.jpg" width=270 height=200></td>
  </tr>
 </table>

 # Instructions

Note, python==3.9.7 version was used, on others something might not work. 

To train previous/new models for emotion classification:

1) Download fer2013 dataset
2) Unpack it in root directory of project
3) Run emotion_recognition notebook with your model
4) Save model

To run real-time emotion demo you will simply need to install requirements and clone repo after that run 
> python3 main.py


