import tensorflow as tf
import cv2
import os
import numpy as np



new_model=tf.keras.models.load_model("trained_model.h5")
import cv2
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascPath)
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    success,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        facess=faceCascade.detectMultiScale(roi_gray)
        if(len(facess)==0):
            print("face not detected")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi=roi_color[ey:ey+eh,ex:ex+ew]
                final_image=cv2.resize(face_roi,(224,224))
                final_image=np.expand_dims(final_image,axis=0)
                font=cv2.FONT_HERSHEY_SIMPLEX
                Predictions=new_model.predict(final_image)
                font_scale=1.5
                if(np.argmax(Predictions)==0):
                 emo="ANGRY"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                elif(np.argmax(Predictions)==1):
                 emo="DISGUST"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                if(np.argmax(Predictions)==2):
                 emo="FEAR"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                if(np.argmax(Predictions)==3):
                 emo="HAPPY"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                if(np.argmax(Predictions)==4):
                 emo="NEUTRAL"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                if(np.argmax(Predictions)==5):
                 emo="SAD"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
                elif(np.argmax(Predictions)==6):
                 emo="SURPRISE"
                 x1,y1,w1,h1=0,0,175,175
                 cv2.rectangle(img,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
                 cv2.putText(img,emo,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                 cv2.putText(img,emo,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255))
        cv2.imshow("Face Emotion Recognition",img)
        if(cv2.waitKey(100)==ord("q")):
            break
cap.release()
cv2.destroyAllWindows()

