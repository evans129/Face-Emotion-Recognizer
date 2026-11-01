import streamlit as st
import cv2
import os
import threading
from PIL import Image
import  numpy as np
from ai_edge_litert.interpreter import Interpreter
import yagmail


class TFLiteModel:
    """Runs a TensorFlow Lite model through the same predict() call the Keras model used."""

    def __init__(self, path):
        self._interpreter = Interpreter(model_path=path)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._lock = threading.Lock()

    def predict(self, batch):
        samples = np.asarray(batch, dtype=np.float32).reshape((-1, *self._input["shape"][1:]))
        outputs = []
        # Streamlit runs each session on its own thread, and an interpreter isn't thread-safe.
        with self._lock:
            for sample in samples:
                self._interpreter.set_tensor(self._input["index"], sample[np.newaxis])
                self._interpreter.invoke()
                outputs.append(self._interpreter.get_tensor(self._output["index"])[0].copy())
        return np.array(outputs)


@st.cache_resource
def load_model(path):
    return TFLiteModel(path)


def email_credentials():
    """Sender address and Gmail app password from Streamlit secrets, or None when email isn't set up."""
    try:
        sender = st.secrets["email"]
        return sender["user"], sender["app_password"]
    except (KeyError, FileNotFoundError):
        return None


caspath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(caspath)
new_model=load_model("trained_model.tflite")
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
    <div style="background-color:#ff6f61;padding:10px">
    <h2 style="color:white;text-align:center;">Emotion recognition app</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    emo=''
    image = None
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.text("original Image")
        st.image(image)
    else:
     image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
     if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        if image is None:
            st.warning('Take a picture or upload an image first', icon="⚠️")
        else:
            result_img, emotion=detect_faces(image,emo)
            dat = "The Uploaded Face is " + emotion
            st.image(result_img)
            st.download_button("Download Analysis", dat, file_name='Processed_Image.txt', key='Download Image Analysis')
    credentials = email_credentials()
    if credentials is None:
        st.caption("Emailing results is turned off in this demo. Use Download Analysis instead.")
        return
    rec = st.text_input('Enter your email')
    if st.button("Mail to me"):
          if image is None:
           st.warning('Take a picture or upload an image first', icon="⚠️")
          elif(len(rec)!=0):
           result_img, emotion = detect_faces(image, emo)
           dat = "The Uploaded Face is " + emotion
           yag = yagmail.SMTP(*credentials)
           contents = [dat]
           yag.send(rec, 'subject', contents)
          else:
           st.warning('Please Enter Your Email', icon="⚠️")
if __name__== '__main__':
    main()
