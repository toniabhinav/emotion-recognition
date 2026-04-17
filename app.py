from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

model = load_model("emotion_model.hdf5")

labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_counts = {label: 0 for label in labels}
emotion_history = []

@app.route('/')
def home():
    return render_template("index.html")


# 🔥 DETECTION
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])

    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({
            "emotion": "No Face",
            "confidence": 0,
            "history": emotion_history
        })

    (x, y, w, h) = faces[0]

    roi = gray[y:y+h, x:x+w]
    roi = cv2.equalizeHist(roi)
    roi = cv2.resize(roi, (48,48)) / 255.0
    roi = np.reshape(roi, (1,48,48,1))

    pred = model.predict(roi, verbose=0)
    confidence = float(np.max(pred))
    emotion = labels[np.argmax(pred)]

    # 📊 Count
    emotion_counts[emotion] += 1

    # 🧠 History (last 10)
    emotion_history.append(emotion)
    if len(emotion_history) > 10:
        emotion_history.pop(0)

    # 🧾 Log file
    with open("emotion_log.txt", "a") as f:
        f.write(f"{datetime.now()} | {emotion} | {round(confidence,2)}\n")

    return jsonify({
        "emotion": emotion,
        "confidence": round(confidence, 2),
        "history": emotion_history
    })


# 📊 GRAPH
@app.route('/graph')
def graph():
    import matplotlib.pyplot as plt

    emotions = []
    counts = []

    for k, v in emotion_counts.items():
        if v > 0:
            emotions.append(k)
            counts.append(v)

    if len(counts) == 0:
        return jsonify({"status": "no data"})

    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%')
    plt.title("Emotion Distribution")

    plt.savefig("static/graph.png")
    plt.close()

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)