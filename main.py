import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]

movie_genre_list = ["Happy", "Sad"]
genre = movie_genre_list[0]  # 장르 설정

i = 0

emotion_value_list = {
    "angry": 0,
    "disgusting": 0,
    "fearful": 0,
    "happy": 0,
    "sad": 0,
    "surprising": 0,
    "neutral": 0,
}

camera = cv2.VideoCapture(0)

while True:
    # Capture image from camera
    ret, frame = camera.read()

    # Convert color to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection in frame
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))

    # Create empty image
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # Perform emotion recognition only when face is detected
    if len(faces) > 0:
        # For the largest image
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Emotion predict
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # print("==========")
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # print(f"emotion: {emotion}, prob: {prob}, w: {w}")
            # print("==========")

            current_emotion = ""

            if emotion == "Angry":
                current_emotion = "angry"
            # elif emotion == "Disgusting":
            #   current_emotion = "disgusting"
            elif emotion == "Fearful":
                current_emotion = "fearful"
            elif emotion == "Happy":
                current_emotion = "happy"
            elif emotion == "Sad":
                current_emotion = "sad"
            elif emotion == "Surprising":
                current_emotion = "surprising"
            # elif emotion == "Neutral":
            #   current_emotion = "neutral"

            if not current_emotion == "":
                emotion_value_list[current_emotion] += prob * 100

    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

most_emotion = max(emotion_value_list, key=emotion_value_list.get)

for k, v in emotion_value_list.items():
    emotion_value_list[k] = round(emotion_value_list[k] / i, 2)

print(emotion_value_list)
print(f"Max Value: {most_emotion}")


def make_result_text(inner_result_text, max_emotion_text):
    result_image = np.full((300, 300, 3), (255, 255, 255), np.uint8)

    y0, dy = 50, 30

    for k, line in enumerate(inner_result_text.split("\n")):
        y = y0 + k*dy
        cv2.putText(result_image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(result_image, max_emotion_text, (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("result", result_image)
    cv2.waitKey()


result_text = f"Angry: {emotion_value_list['angry']} \nFearful: {emotion_value_list['fearful']} \nHappy: {emotion_value_list['happy']}\n Sad: {emotion_value_list['sad']} \nSurprising: {emotion_value_list['surprising']}"

make_result_text(result_text, f"Max Emotion: {most_emotion}")
