
#!/usr/bin/env python
"""object_counter.py — Live YOLO object counter from webcam."""
import cv2
from ultralytics import YOLO
from collections import Counter
import time

model = YOLO('yolo11n.pt')
cap   = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_t = time.time()

for result in model.predict(source=cap, stream=True, conf=0.4, verbose=False):
    frame  = result.plot()
    counts = Counter(model.names[int(b.cls[0])] for b in result.boxes)

    # FPS
    now   = time.time()
    fps   = 1 / (now - prev_t + 1e-6)
    prev_t = now

    # Overlay
    cv2.putText(frame, f'FPS: {fps:.1f}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    for i,(cls,cnt) in enumerate(sorted(counts.items())):
        cv2.putText(frame, f'{cls}: {cnt}', (10, 65+i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow('Object Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()
