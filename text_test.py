import cv2
import numpy as np


def make_result_text(text, max_emotion_text):
    result_image = np.full((300, 300, 3), (255, 255, 255), np.uint8)

    y0, dy = 50, 30

    for k, line in enumerate(text.split("\n")):
        y = y0 + k*dy
        cv2.putText(result_image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(result_image, max_emotion_text, (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("result", result_image)
    cv2.waitKey()
