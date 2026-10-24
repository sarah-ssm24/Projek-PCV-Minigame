import cv2, numpy as np, pygame, math

pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Cooking Color - V4")
clock = pygame.time.Clock()
SMALL = pygame.font.SysFont("arial", 20)
WHITE, YELLOW = (255, 255, 255), (255, 215, 50)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Kamera tidak dapat dibuka.")

LOW, HIGH = np.array([18, 80, 80]), np.array([40, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)
areas = []

items = [dict(name=n, color=c, x=x, y=300)
         for n, c, x in (("EGG", (255, 245, 220), 170), ("TOMATO", (220, 70, 70), 300), ("CHEESE", (255, 190, 40), 430))]
holding = None


def detect(frame):
    """Return (x, y, pinch) dalam koordinat kamera, atau None."""
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), LOW, HIGH)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL), cv2.MORPH_CLOSE, KERNEL)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if area < 800 or M["m00"] == 0:
        return None
    hull = cv2.contourArea(cv2.convexHull(c))
    areas.append(area)
    del areas[:-15]
    ratio = area / (np.mean(areas) + 1)
    pinch = len(areas) >= 5 and 0.65 < ratio < 1.10 and (area / hull if hull else 1) > 0.60
    return M["m10"] / M["m00"], M["m01"] / M["m00"], pinch


running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False

    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)  # sudah mirror, jadi langsung diskalakan ke layar
    r = detect(frame)
    hand = (int(r[0] / frame.shape[1] * W), int(r[1] / frame.shape[0] * H)) if r else None
    pinch = bool(r and r[2])

    # --- logika
    if hand:
        if holding is None:
            if pinch:
                for it in items:
                    if math.dist(hand, (it["x"], it["y"])) < 80:
                        holding = it
                        break
        elif pinch:
            holding["x"], holding["y"] = hand
        else:
            holding = None

    # --- draw
    screen.fill((250, 230, 205))
    pygame.draw.rect(screen, (190, 130, 85), (0, 470, W, 250))
    for it in items:
        x, y = int(it["x"]), int(it["y"])
        pygame.draw.circle(screen, (100, 70, 60), (x + 5, y + 8), 48)
        pygame.draw.circle(screen, it["color"], (x, y), 45)
        lbl = SMALL.render(it["name"], True, (45, 40, 50))
        screen.blit(lbl, (x - lbl.get_width() // 2, y + 55))
    if hand:
        pygame.draw.circle(screen, YELLOW, hand, 18, 4)
        screen.blit(SMALL.render("PINCH" if pinch else "HAND", True, (45, 40, 50)), (hand[0] + 25, hand[1] - 15))
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()