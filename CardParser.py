import cv2
import numpy as np
import Cards
import os
from picamera.array import PiRGBArray
from picamera import PiCamera

images_location = os.path.dirname(os.path.abspath(__file__)) + '/Card_Imgs/'

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

pi_camera = PiCamera()
pi_camera.resolution = (IMAGE_WIDTH, IMAGE_HEIGHT)
pi_camera.framerate = 10
raw_capture = \
    PiRGBArray(
        pi_camera,
        size=(IMAGE_WIDTH, IMAGE_HEIGHT)
    )

i = 1

for Name in ['Ace','Two','Three','Four','Five','Six','Seven','Eight',
             'Nine','Ten','Jack','Queen','King','Spades','Diamonds',
             'Clubs','Hearts']:

    filename = Name + '.jpg'

    print('Press "p" to take a picture of ' + filename)

    raw_capture.truncate(0)
    for frame in pi_camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Card", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            break
        raw_capture.truncate(0)
    gray = \
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, threshold = \
        cv2.threshold(
            blur,
            100,
            255,
            cv2.THRESH_BINARY
        )

    contours, hierarchy = \
        cv2.findContours(
            threshold,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

    contours = \
        sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

    flag = 0
    image2 = image.copy()

    if len(contours) == 0:
        print('No contours found!')
        quit()

    card = contours[0]

    perimeter = cv2.arcLength(card, True)
    a = \
        cv2.approxPolyDP(
            card,
            0.01 * perimeter,
            True
        )
    pts = np.float32(a)

    x, y, w, h = cv2.boundingRect(card)

    warp = \
        Cards.flattener(
            image,
            pts,
            w,
            h
        )

    corner = warp[0:138, 0:49]
    corner_zoom = \
        cv2.resize(
            corner,
            (0,0),
            fx=4,
            fy=4
        )
    corner_blur = \
        cv2.GaussianBlur(
            corner_zoom,
            (5, 5),
            0
        )
    _, corner_threshold = \
        cv2.threshold(
            corner_blur,
            155,
            255,
            cv2.THRESH_BINARY_INV
        )

    if i <= 13:
        rank = corner_threshold[0:300, 0:200]
        cv2.imshow("Rank", rank)
        rank_contours, hierarchy =\
            cv2.findContours(
                rank,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
        rank_contours = \
            sorted(
                rank_contours,
                key=cv2.contourArea,
                reverse=True
            )
        x, y, w, h = cv2.boundingRect(rank_contours[0])
        final_img = \
            cv2.resize(
                rank[y : y + h, x : x + w],
                (RANK_WIDTH, RANK_HEIGHT),
                0,
                0
            )

    if i > 13: # Isolate suit
        suit = corner_threshold[290:, 0:200]
        cv2.imshow("Suite", suit)
        suit_contours, hierarchy =\
            cv2.findContours(
                suit,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
        suit_contours =\
            sorted(
                suit_contours,
                key=cv2.contourArea,
                reverse=True
            )
        x, y, w, h = cv2.boundingRect(suit_contours[0])
        final_img = \
            cv2.resize(
                suit[y : y + h, x : x + w],
                (SUIT_WIDTH, SUIT_HEIGHT),
                0,
                0
            )


    cv2.imshow("Result",final_img)
    cv2.imshow("Card",warp)
    cv2.imshow("Corner_zoom", corner_zoom)

    print('Press "c" to continue.')
    key = cv2.waitKey(0) & 0xFF
    if key == ord('c'):
        cv2.imwrite(images_location + filename, final_img)

    i = i + 1

cv2.destroyAllWindows()
pi_camera.close()
