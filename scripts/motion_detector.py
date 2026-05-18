
#!/usr/bin/env python
"""motion_detector.py — Webcam motion detection with alert overlay."""
import cv2, time

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    fgmask  = fgbg.apply(blurred)
    fgmask  = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    motion = False
    for cnt in contours:
        if cv2.contourArea(cnt) < 1500: continue
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        motion = True

    label = '⚠ MOTION DETECTED' if motion else 'CLEAR'
    color = (0,0,255) if motion else (0,200,0)
    cv2.putText(frame, label, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, time.strftime(' %H:%M:%S'), (10,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow('Motion Detector', frame)
    cv2.imshow('FG Mask', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
