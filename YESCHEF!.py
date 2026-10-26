import cv2, numpy as np, pygame, math, os

ASSET_DIR = r"C:\Users\Sarah SM\Documents\PCV\Projek 1\assets"
FILES = ["pot.png", "mie.png", "egg.png", "bumbu.png"]

START_SCORE = 100
PER_ITEM = 100
GRACE = 10
PENALTY = 10
LOST_MAX = 20

missing = [f for f in FILES if not os.path.isfile(os.path.join(ASSET_DIR, f))]
if missing:
    raise SystemExit(f"Aset tidak ditemukan di {ASSET_DIR}:\n  " + "\n  ".join(missing))
os.chdir(ASSET_DIR)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Kamera tidak dapat dibuka.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("YESCHEF!")
clock = pygame.time.Clock()

SMALL, MED, BIG = (
    pygame.font.SysFont("arial", s, bold=b)
    for s, b in ((20, 0), (28, 1), (42, 1))
)

YELLOW, DARK, WHITE, RED = (
    (255, 215, 50),
    (45, 40, 50),
    (255, 255, 255),
    (230, 80, 80)
)

LOW, HIGH = np.array([0, 140, 80]), np.array([255, 173, 127])
KERNEL = np.ones((5, 5), np.uint8)
GAP = 1
hist = []
last_mask = np.zeros((480, 640), np.uint8)
show_mask = True

FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
) if hasattr(cv2, "CascadeClassifier") else None

faces, fcount = [], 0

SIZE = 100
scale = pygame.transform.smoothscale

pot = scale(pygame.image.load("pot.png"), (260, 190))
mie = scale(pygame.image.load("mie.png"), (SIZE, SIZE))
telur = scale(pygame.image.load("egg.png"), (SIZE, SIZE))
bumbu = scale(pygame.image.load("bumbu.png"), (SIZE, SIZE))

POT = (1000, 440)


def reset():
    global items, holding, state, lost, t0, secs

    items = [
        dict(name=n, img=i, x=x, y=300, done=False)
        for n, i, x in (
            ("Mie", mie, 170),
            ("Telur", telur, 300),
            ("Bumbu", bumbu, 430)
        )
    ]

    holding, state, lost, secs = None, "PLAY", 0, 0
    t0 = pygame.time.get_ticks()


def draw_bg():
    screen.fill((250, 230, 205))
    pygame.draw.rect(screen, (255, 220, 190), (0, 0, W, 470))
    pygame.draw.rect(screen, (190, 130, 85), (0, 470, W, 250))
    pygame.draw.rect(screen, (255, 170, 100), (0, 0, W, 75))
    screen.blit(BIG.render("YESCHEF!", True, WHITE), (35, 18))


def text(s, x, y, font=SMALL, color=WHITE, center=False):
    img = font.render(s, True, color)
    screen.blit(
        img,
        (x - img.get_width() // 2 if center else x, y)
    )


def detect(frame):
    global faces, fcount, last_mask

    fcount += 1

    if fcount % 5 == 1 and FACE is not None and not FACE.empty():
        f = FACE.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            1.2,
            5,
            minSize=(60, 60)
        )

        if len(f):
            faces = f

    mask = cv2.inRange(
        cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb),
        LOW,
        HIGH
    )

    for x, y, w, h in faces:
        mask[
            max(0, y - h // 4): y + int(h * 1.6),
            max(0, x - w // 4): x + w + w // 4
        ] = 0

    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL),
        cv2.MORPH_CLOSE,
        KERNEL
    )

    last_mask = mask

    cnts, hier = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    outer = [
        i
        for i, h in enumerate(hier[0] if cnts else [])
        if h[3] == -1
    ]

    if not outer:
        hist.clear()
        return None

    i = max(outer, key=lambda k: cv2.contourArea(cnts[k]))
    c, area = cnts[i], cv2.contourArea(cnts[i])

    if area < 800:
        hist.clear()
        return None

    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [c], -1, 255, -1)

    _, radius, _, (px, py) = cv2.minMaxLoc(
        cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    )

    try:
        d = cv2.convexityDefects(
            c,
            cv2.convexHull(c, returnPoints=False)
        )
    except cv2.error:
        d = None

    gaps = (
        0
        if d is None
        else int(
            (
                d.reshape(-1, 4)[:, 3] / 256 > GAP * radius
            ).sum()
        )
    )

    ring = any(
        h[3] == i and cv2.contourArea(cnts[k]) > 0.04 * area
        for k, h in enumerate(hier[0])
    )

    hist.append(gaps < 2 or ring)
    del hist[:-7]

    return px, py, sum(hist) >= 4


reset()
sm = None

try:
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (
                e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
            ):
                running = False

            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                reset()

            elif e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                show_mask = not show_mask

        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        r = detect(frame)

        if r:
            raw = (
                r[0] / frame.shape[1] * W,
                r[1] / frame.shape[0] * H
            )

            sm = (
                raw
                if sm is None
                else (
                    sm[0] * 0.5 + raw[0] * 0.5,
                    sm[1] * 0.5 + raw[1] * 0.5
                )
            )
        else:
            sm = None

        hand = (int(sm[0]), int(sm[1])) if sm else None
        pinch = bool(r and r[2])

        if state == "PLAY":
            secs = (pygame.time.get_ticks() - t0) / 1000

            if hand:
                lost = 0

                if holding is None:
                    if pinch:
                        for it in items:
                            if (
                                not it["done"]
                                and math.dist(
                                    hand,
                                    (it["x"], it["y"])
                                ) < SIZE
                            ):
                                holding = it
                                break

                elif pinch:
                    holding["x"], holding["y"] = hand

                else:
                    if math.dist(hand, POT) < 130:
                        holding["done"] = True
                    else:
                        state = "OVER"

                    holding = None

            elif holding:
                lost += 1

                if lost > LOST_MAX:
                    state = "OVER"

            if all(i["done"] for i in items):
                state = "WIN"

        n_done = sum(i["done"] for i in items)

        score = max(
            0,
            START_SCORE
            + PER_ITEM * n_done
            - PENALTY * int(max(0, secs - GRACE))
        )

        draw_bg()

        screen.blit(pot, pot.get_rect(center=POT))

        for it in items:
            if not it["done"]:
                x, y = int(it["x"]), int(it["y"])

                screen.blit(
                    it["img"],
                    it["img"].get_rect(center=(x, y))
                )

                text(
                    it["name"],
                    x,
                    y + SIZE // 2 + 10,
                    SMALL,
                    DARK,
                    True
                )

        pygame.draw.rect(
            screen,
            WHITE,
            (920, 20, 320, 55),
            border_radius=15
        )

        text(
            f"SCORE: {score}",
            945,
            34,
            SMALL,
            DARK
        )

        text(
            f"TIME {int(secs)}s",
            700,
            28,
            MED,
            RED if secs > GRACE else WHITE
        )

        text(
            f"Bahan {n_done}/{len(items)}",
            480,
            32,
            SMALL
        )

        if state == "PLAY" and secs > GRACE:
            text(
                f"Kelamaan! -{PENALTY} poin/detik",
                55,
                640,
                MED,
                RED
            )

        text(
            "Genggam tangan = ambil bahan  |  Buka tangan di atas panci = masukkan",
            55,
            685,
            SMALL
        )

        if show_mask:
            m = cv2.cvtColor(
                cv2.resize(last_mask, (192, 144)),
                cv2.COLOR_GRAY2RGB
            )

            screen.blit(
                pygame.image.frombuffer(
                    m.tobytes(),
                    (192, 144),
                    "RGB"
                ),
                (20, 90)
            )

            pygame.draw.rect(
                screen,
                WHITE,
                (20, 90, 192, 144),
                2
            )

        if hand:
            pygame.draw.circle(
                screen,
                YELLOW,
                hand,
                18,
                4
            )

            text(
                "PINCH" if pinch else "HAND",
                hand[0] + 25,
                hand[1] - 15,
                SMALL,
                DARK
            )

        if state != "PLAY":
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((30, 20, 30, 210))
            screen.blit(ov, (0, 0))

            text(
                "GAME OVER" if state == "OVER" else "MASAKAN SELESAI!",
                W // 2,
                230,
                BIG,
                RED if state == "OVER" else YELLOW,
                True
            )

            if state == "OVER":
                text(
                    "Bahannya jatuh!",
                    W // 2,
                    295,
                    MED,
                    WHITE,
                    True
                )

            text(
                f"SCORE: {score}   TIME: {int(secs)}s",
                W // 2,
                340,
                MED,
                WHITE,
                True
            )

            text(
                "Tekan R untuk main lagi",
                W // 2,
                400,
                SMALL,
                WHITE,
                True
            )

        pygame.display.flip()
        clock.tick(60)

finally:
    cap.release()
    pygame.quit()