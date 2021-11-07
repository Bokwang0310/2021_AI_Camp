import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]

score = 0
count = 0

emotion_value_list = {
    "Angry": 0,
    "Disgusting": 0,
    "Fearful": 0,
    "Happy": 0,
    "Sad": 0,
    "Surprising": 0,
    "Neutral": 0,
}

emotion_score = {"Angry": 5, "Disgusting": 0, "Fearful": 5, "Happy": 5, "Sad": 5, "Neutral": 0, "Surprising": 5}
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

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            if emotion == "Neutral":
                if w > 300 and count % 2 == 0:
                    score += emotion_score[emotion]
                    count += 1
            else:
                if emotion in emotion_score and w > 25:
                    score += emotion_score[emotion]
                    count += 1

            print(f"emotion: {emotion}, prob: {prob}, w: {w}")
            print(score)

            emotion_value_list[emotion] += prob * 100

    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

most_emotion = max(emotion_value_list, key=emotion_value_list.get)

percentage_list = {}

for k, v in emotion_value_list.items():
    percentage_list[k] = round(emotion_value_list[k] / i, 2)

print(percentage_list)
print(f"Max Value: {most_emotion}")


def make_result_text(inner_result_text, max_emotion_text, movie_score):
    result_image = np.full((330, 310, 3), (255, 255, 255), np.uint8)

    y0, dy = 50, 30

    for ii, line in enumerate(inner_result_text.split("\n")):
        y = y0 + ii*dy
        cv2.putText(result_image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(result_image, max_emotion_text, (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(result_image, f"Score: {movie_score}", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    while True:
        cv2.imshow("result", result_image)
        key = cv2.waitKey(33)

        # detect q
        if key == 113:
            break
        else:
            continue


result_text = f"Angry: {percentage_list['Angry']} \nFearful: {percentage_list['Fearful']} \nHappy: {percentage_list['Happy']}\n Sad: {percentage_list['Sad']} \nSurprising: {percentage_list['Surprising']} \nBoring: {percentage_list['Neutral']}"

make_result_text(result_text, f"Max Emotion: {most_emotion}", round(score / count, 1))

cv2.destroyAllWindows()

print(f"Total Score: {score}, Total Count: {count}")
print(f"Final Score: {round(score / count, 1)}")
