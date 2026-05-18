
#!/usr/bin/env python
"""doc_scanner.py — Live document scanner from webcam."""
import cv2, numpy as np

def order_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
    return rect

def warp(img, pts):
    rect=(tl,tr,br,bl)=order_points(pts)
    W=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl)))
    H=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],dtype=np.float32)
    return cv2.warpPerspective(img,cv2.getPerspectiveTransform(rect,dst),(W,H))

cap = cv2.VideoCapture(0)
scanned = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    display = frame.copy()
    gray=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(5,5),0)
    edges=cv2.dilate(cv2.Canny(gray,50,150),np.ones((3,3),np.uint8))
    cnts,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
    for cnt in cnts:
        approx=cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        if len(approx)==4:
            cv2.drawContours(display,[approx],-1,(0,255,0),3)
            scanned=warp(frame,approx.reshape(4,2).astype(np.float32))
            break
    cv2.putText(display,'[S]=save  [Q]=quit',(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.imshow('Scanner',display)
    if scanned is not None: cv2.imshow('Scan Preview',scanned)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    if key==ord('s') and scanned is not None:
        cv2.imwrite('scan_output.jpg',scanned)
        print('Saved scan_output.jpg')

cap.release(); cv2.destroyAllWindows()
