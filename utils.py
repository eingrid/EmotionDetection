import numpy as np

from retinaface import RetinaFace
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2 


prev_emotion = 'neutral'
second_emotion = 'neutral'


prev_confidance = 0
second_percentage = 0

def get_faces(img,method_name = 'opencv'):
    """
    Returns list of faces described by four coords.
    """
    res = []
    if method_name == 'opencv':
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
        face_cascade = cv2.CascadeClassifier('opencv_detector/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 3)
        for x, y, w, h in faces:
            if w*h > 5000:
                res.append([x, y, x+w, y+h])

    elif  method_name == 'retinaface':
        faces = RetinaFace.detect_faces(img)
        for face in faces:
          x, y, x2, y2 = faces[face]['facial_area']
          res.append([x, y, x2, y2])
    elif method_name =='mtcnn':
        detector = MTCNN()
        faces = detector.detect_faces(img)
        for face in faces:
            x, y, x2, y2 = face['box']
        res.append([x, y, x+x2, y+y2])
    else:
        raise NameError('Wrong method_name in get_faces function.')

    return res      
        
def build_model(model_name = 'xception'):
    """
    Loads model and returns it as well as input size of a model.

    Parameters
        ----------
        model_name : str
            The name of the model.
        augmented_model : bool, optional
            Whether to use a model trained on augmented dataset.
        finetuned : bool, optional
            
    """
    if model_name == 'xception': 
        model = load_model('models/raw_xception.hdf5')
        model.load_weights('models/xception.h5')
        return model, ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], (64,64)           
 
    elif model_name == 'scratch': 
        model = load_model('models/conv_net')
        return model,['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],(48,48)
    else:
        raise NameError('Wrong method_name in get_faces function.')
        
def draw_faces(model, img, num, method_name, target_size, emotion_labels): 
    """
    Returns a image with detected facial expressions.
    """
    global prev_confidance;
    global prev_emotion;
    global second_emotion;
    global second_percentage;

    
    faces = get_faces(img,method_name)
    for x,y,w,h in faces:
      face_crop = img[y:h,x:w]
      
      img_resized = cv2.resize(face_crop, target_size)


      gray2 = cv2.cvtColor(img_resized,cv2.COLOR_RGB2GRAY)/255.0
      gray2 = np.expand_dims(np.expand_dims(gray2, -1), 0)

      if num % 10 == 0:
        
        prediction = model.predict(gray2,verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        second_emotion = emotion_labels[np.argsort(np.max(prediction, axis=0))[-2]]

        percentage = int(round(np.max(prediction),2) * 100)
        prediction.sort()
        second_percentage = int(prediction[0][-2] * 100)
        prev_confidance = percentage

        prev_emotion = emotion

        img = cv2.putText(img,f'{emotion}:{percentage}%, {second_emotion}:{second_percentage}%',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
      else:
            
        img = cv2.putText(img,f'{prev_emotion}:{prev_confidance}%, {second_emotion}:{second_percentage}%',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
              

      img = cv2.rectangle(img,(x,y),(w,h),(255,0,0),2)
    return img

