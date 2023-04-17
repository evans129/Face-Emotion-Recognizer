import streamlit as st
import cv2
import os
from PIL import Image
import  numpy as np
import tensorflow as tf
import yagmail
caspath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(caspath)
new_model=tf.keras.models.load_model("trained_model.h5")
#new_model=cv2.face.LBPHFaceRecognizer_create()
def detect_faces(image,emo):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
     roi_gray = gray[y:y + h, x:x + w]
     roi_color = img[y:y + h, x:x + w]
     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     facess = face_cascade.detectMultiScale(roi_gray)
     if (len(facess) == 0):
        print("face not detected")
     else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey:ey + eh, ex:ex + ew]
            final_image = cv2.resize(face_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            Predictions = new_model.predict(final_image)
            font_scale = 1.5
            if (np.argmax(Predictions) == 0):
                emo = "ANGRY"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            elif (np.argmax(Predictions) == 1):
                emo = "DISGUST"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            if (np.argmax(Predictions) == 2):
                emo = "FEAR"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            if (np.argmax(Predictions) == 3):
                emo = "HAPPY"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            if (np.argmax(Predictions) == 4):
                emo = "NEUTRAL"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            if (np.argmax(Predictions) == 5):
                emo = "SAD"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))
            elif (np.argmax(Predictions) == 6):
                emo = "SURPRISE"
                x1, y1, w1, h1 = 0, 0, 175, 175
                cv2.rectangle(img, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(img, emo, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(img, emo, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))

    return img,emo
def main():
    st.title("Face Emotion Recognizer")
    html_temp="""
    <body style="background-color:red;">
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;">Face recognition app</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    emo=''
    image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        result_img, emotion=detect_faces(image,emo)
        dat = "The Uploaded Face is " + emotion
        st.image(result_img)
        st.download_button("Download Analysis", dat, file_name='Processed_Image.txt', key='Download Image Analysis')
    rec = st.text_input('Enter your email')
    if st.button("Mail to me"):
          result_img, emotion = detect_faces(image, emo)
          dat = "The Uploaded Face is " + emotion
          if(len(rec)!=0):
           yag = yagmail.SMTP('devanshkumaravi@gmail.com', 'oowhmqyyreotkwys')
           contents = [dat]
           yag.send(rec, 'subject', contents)
          else:
           st.warning('Please Enter Your Email', icon="⚠️")
if __name__== '__main__':
    main()
