
#!/usr/bin/env python
"""face_blur.py — Real-time face anonymisation from webcam."""
import cv2, urllib.request, os

CASCADE_PATH = 'haarcascade_frontalface_default.xml'
if not os.path.exists(CASCADE_PATH):
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/' + CASCADE_PATH
    urllib.request.urlretrieve(url, CASCADE_PATH)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap = cv2.VideoCapture(0)

blur_mode = True   # toggle with SPACE

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

    if blur_mode:
        for (x, y, w, h) in faces:
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(
                frame[y:y+h, x:x+w], (55,55), 30)

    cv2.putText(frame, f'Faces: {len(faces)}  [SPACE]=toggle  [Q]=quit',
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow('Face Blur', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord(' '): blur_mode = not blur_mode

cap.release()
cv2.destroyAllWindows()
