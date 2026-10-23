import cv2, numpy as np

LOW, HIGH = np.array([18, 80, 80]), np.array([40, 255, 255])  
KERNEL = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Kamera tidak dapat dibuka.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LOW, HIGH)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 800:  # abaikan noise
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    cv2.imshow("Kamera", frame)
    cv2.imshow("Mask kuning", mask)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
