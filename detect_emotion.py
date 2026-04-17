import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

model = load_model('emotion_model.hdf5')

labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 🔥 SMOOTHING (prevents flickering)
emotion_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]

        # 🔥 IMPORTANT PREPROCESSING
        roi = cv2.equalizeHist(roi)
        roi = cv2.resize(roi,(48,48)) / 255.0
        roi = np.reshape(roi,(1,48,48,1))

        prediction = model.predict(roi, verbose=0)
        confidence = np.max(prediction)

        if confidence > 0.5:
            emotion = labels[np.argmax(prediction)]
        else:
            emotion = "Uncertain"

        emotion_buffer.append(emotion)
        final_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, f"{final_emotion} ({confidence:.2f})",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,
                    (0,255,0),2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()