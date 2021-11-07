import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# 얼굴 인식 및 감정 분류 파일 불러오기
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)

# 측정할 감정 리스트
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]

# 감정들의 값을 저장할 딕셔너리
emotion_value_list = {
    "Angry": 0,
    "Disgusting": 0,
    "Fearful": 0,
    "Happy": 0,
    "Sad": 0,
    "Surprising": 0,
    "Neutral": 0,
}

# 각 감정에 대한 점수
emotion_score = {"Angry": 5, "Disgusting": 0, "Fearful": 5, "Happy": 5, "Sad": 5, "Neutral": 0, "Surprising": 5}

score = 0
count = 0

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 프레임 속 얼굴 인식
    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30))

    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # 얼굴이 인식되었을 때 감정 판단 수행
    if len(faces) > 0:
        # 인식된 얼굴 중 가장 큰 이미지 선택
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        
        # 학습에 이용하기 위해 얼굴 사진을 48*48 사이즈로 변경
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 감정 예측
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # 라벨링
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # Neutral에 대해서는 가중치를 낮게 하기 위해 값을 높은 것만 잡고 횟수도 거름
            if emotion == "Neutral":
                if w > 200 and count % 2 == 0:
                    score += emotion_score[emotion]
                    count += 1
            # 다른 감정들에 대해서도 가중치 선별
            else:
                if emotion in emotion_score and w > 25:
                    score += emotion_score[emotion]
                    count += 1

            print(f"Current Emotion: {emotion}, prob: {prob}, w: {w}")
            print(f"Current Score: {score}")

            # 사전에 만든 딕셔너리에 감정값 추가
            emotion_value_list[emotion] += prob * 100

    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# 가장 높은 값을 보인 감정
most_emotion = max(emotion_value_list, key=emotion_value_list.get)

# 퍼센테이지 변환
percentage_list = {}
for k, v in emotion_value_list.items():
    percentage_list[k] = round(emotion_value_list[k] / count, 2)

print(percentage_list)
print(f"Max Value Emotion: {most_emotion}")


# 결과 박스 띄우기
def make_result_text(inner_result_text, max_emotion_text, movie_score):
    result_image = np.full((330, 310, 3), (255, 255, 255), np.uint8)

    y0, dy = 50, 30

    for ii, line in enumerate(inner_result_text.split("\n")):
        y = y0 + ii*dy
        cv2.putText(result_image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(result_image, max_emotion_text, (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(result_image, movie_score, (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    while True:
        cv2.imshow("result", result_image)
        key = cv2.waitKey(33)

        # q 감지
        if key == 113:
            break
        else:
            continue


result_text = f"Angry: {percentage_list['Angry']} \nFearful: {percentage_list['Fearful']} \nHappy: {percentage_list['Happy']} \nSad: {percentage_list['Sad']} \nSurprising: {percentage_list['Surprising']} \nBoring: {percentage_list['Neutral']}"

make_result_text(result_text, f"Max Emotion: {most_emotion}", f"Score: {round(score / count, 1)}")

cv2.destroyAllWindows()

print(f"Total Score: {score}, Total Count: {count}")
print(f"Final Score: {round(score / count, 1)}")
