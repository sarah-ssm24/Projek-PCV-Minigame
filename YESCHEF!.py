import cv2, numpy as np

LOW, HIGH = np.array([18, 80, 80]), np.array([40, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)
areas = []


def detect(frame):
    """Return (hand dict atau None, mask). hand["pinch"] True kalau bentuk tangan menyempit."""
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LOW, HIGH)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if area < 800 or M["m00"] == 0:
        return None, mask

    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull if hull else 1
    areas.append(area)
    del areas[:-15]
    ratio = area / (np.mean(areas) + 1)
    pinch = len(areas) >= 5 and 0.65 < ratio < 1.10 and solidity > 0.60

    hand = dict(x=int(M["m10"] / M["m00"]), y=int(M["m01"] / M["m00"]), bbox=cv2.boundingRect(c),
                area=area, solidity=solidity, pinch=pinch, contour=c)
    return hand, mask


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
    hand, mask = detect(frame)

    if hand:
        x, y, w, h = hand["bbox"]
        color = (0, 255, 255) if hand["pinch"] else (0, 255, 0)
        cv2.drawContours(frame, [hand["contour"]], -1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame, (hand["x"], hand["y"]), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"{'PINCH' if hand['pinch'] else 'HAND'} solidity={hand['solidity']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Kamera", frame)
    cv2.imshow("Mask kuning", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()