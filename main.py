import cv2 

from utils import build_model, draw_faces
import tensorflow as tf

import time
vid = cv2.VideoCapture(0)


method_name ='opencv'
model, emotion_labels, target_size = build_model(model_name='xception') 

frame_number = 0
while(True):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    ret, frame = vid.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start = time.time()
    drawed = draw_faces(model, img, frame_number, method_name, target_size, emotion_labels)
    print(time.time()-start)
    bgr_im = cv2.cvtColor(drawed, cv2.COLOR_RGB2BGR)
    cv2.imshow('predicted_image', bgr_im)
    frame_number+=1

vid.release()
cv2.destroyAllWindows()