import numpy as np
import cv2
import random

BACKGROUND_THRESHOLD_MODIFIER = 60
CARD_THRESHOLD_MODIFIER = 30

# are with rank and suit
CORNER_WIDTH = 49
CORNER_HEIGHT = 138

# placement of rank in corner
RANK_WIDTH = 70
RANK_HEIGHT = 125

# placement of suit in corner
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# maximal acceptable differences
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

# area constraints of cards
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_DUPLEX

# random mapping of colors to suit rank
colors = dict()
for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:
    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        colors[Suit + Rank] = \
            (
                random.randint(30, 255),
                random.randint(30, 255),
                random.randint(30, 255)
            )


class AnalyzedCard:
    def __init__(self):
        self.contour = []
        self.width = 0
        self.height = 0
        self.corners = []
        self.center = []
        self.warp = []
        self.rank_img = []
        self.suit_img = []
        self.best_rank_match = "Unknown"
        self.best_suit_match = "Unknown"
        self.rank_difference = 0
        self.suit_difference = 0
        self.color = (0, 0, 0)
        self.correctness = 0


class TrainedRanks:
    def __init__(self, name, image):
        self.image = image
        self.name = name


class TrainedSuits:
    def __init__(self, name, image):
        self.image = image
        self.name = name


def load_ranks(filepath):
    train_ranks = []
    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(
            TrainedRanks(
                Rank,
                cv2.imread(
                    filepath + Rank + '.jpg',
                    cv2.IMREAD_GRAYSCALE
                )
            )
        )
    return train_ranks


def load_suits(filepath):
    train_suits = []
    for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:
        train_suits.append(
            TrainedSuits(
                Suit,
                cv2.imread(
                    filepath + Suit + '.jpg',
                    cv2.IMREAD_GRAYSCALE
                )
            )
        )
    return train_suits


def prepare_image(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # adaptive threshold
    image_width, image_height = np.shape(image)[:2]
    background_lightning_intensity_level = gray[int(image_height / 100)][int(image_width / 2)]
    thresh_level = background_lightning_intensity_level + BACKGROUND_THRESHOLD_MODIFIER

    return cv2.threshold(
        blur,
        thresh_level,
        255,
        cv2.THRESH_BINARY
    )[1]


def find_cards(threshold_image):
    contours, hierarchy = \
        cv2.findContours(
            threshold_image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

    sorted_indexes = \
        sorted(
            range(
                len(contours)
            ),
            key=lambda i: cv2.contourArea(contours[i]),
            reverse=True
        )

    if len(contours) == 0:
        return [], []

    sorted_contours = []
    sorted_hierarchy = []
    is_contour_card = \
        np.zeros(
            len(contours),
            dtype=int
        )

    for i in sorted_indexes:
        sorted_contours.append(contours[i])
        sorted_hierarchy.append(hierarchy[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(sorted_contours)):
        card_size = cv2.contourArea(sorted_contours[i])
        perimeter = cv2.arcLength(
            sorted_contours[i],
            True
        )
        a = cv2.approxPolyDP(
            sorted_contours[i],
            0.01 * perimeter,
            True
        )

        if (
                (card_size < CARD_MAX_AREA)
                and (card_size > CARD_MIN_AREA)
                and (sorted_hierarchy[i][3] == -1)
                and (len(a) == 4)
        ):
            is_contour_card[i] = 1

    return sorted_contours, is_contour_card


def prepare_card(contour, image):
    card = AnalyzedCard()

    card.contour = contour
    perimeter = cv2.arcLength(contour, True)
    a = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    corners = np.float32(a)
    card.corners = corners

    x, y, width, height = cv2.boundingRect(contour)
    card.width = width
    card.height = height

    average = np.sum(corners, axis=0) / len(corners)
    center_x = int(average[0][0])
    center_y = int(average[0][1])
    card.center = [center_x, center_y]

    card.warp = \
        flattener(
            image,
            corners,
            width,
            height
        )

    zoomed_corner = \
        cv2.resize(
            card.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH],
            (0, 0),
            fx=4,
            fy=4
        )

    sample_pixel = zoomed_corner[15, int((CORNER_WIDTH * 4) / 2)]
    threshold_level = sample_pixel - CARD_THRESHOLD_MODIFIER

    if threshold_level <= 0:
        threshold_level = 1

    _, threshold_corner = \
        cv2.threshold(
            zoomed_corner,
            threshold_level,
            255,
            cv2.THRESH_BINARY_INV
        )

    rank = threshold_corner[0:300, 0:200]
    suit = threshold_corner[290:, 0:200]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    rank_contours, hierarchy = \
        cv2.findContours(
            rank,
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    sorted_rank_contours = \
        sorted(
            rank_contours,
            key=cv2.contourArea, reverse=True
        )

    if len(sorted_rank_contours) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(sorted_rank_contours[0])
        rank_image = \
            cv2.resize(
                rank[y1: y1 + h1, x1: x1 + w1],
                (RANK_WIDTH, RANK_HEIGHT),
                0,
                0
            )
        card.rank_img = rank_image

    suit_contours, hierarchy = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_suit_contours = \
        sorted(
            suit_contours,
            key=cv2.contourArea,
            reverse=True
        )

    if len(suit_contours) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(sorted_suit_contours[0])
        suit_image = \
            cv2.resize(
                suit[y2: y2 + h2, x2: x2 + w2],
                (SUIT_WIDTH, SUIT_HEIGHT),
                0,
                0
            )
        card.suit_img = suit_image

    return card


def match_card(card, sample_ranks, sample_suits):
    best_matched_rank_difference = 10000
    best_matched_suit_difference = 10000
    best_matched_rank_name = "Unknown"
    best_matched_suit_name = "Unknown"

    if (len(card.rank_img) != 0) and (len(card.suit_img) != 0):
        for sample_rank in sample_ranks:

            difference_image = cv2.absdiff(card.rank_img, sample_rank.image)
            rank_difference = int(np.sum(difference_image) / 255)

            if rank_difference < best_matched_rank_difference:
                best_matched_rank_difference = rank_difference
                best_rank_name = sample_rank.name

        for sample_suit in sample_suits:

            difference_image = cv2.absdiff(card.suit_img, sample_suit.image)
            suit_difference = int(np.sum(difference_image) / 255)

            if suit_difference < best_matched_suit_difference:
                best_matched_suit_difference = suit_difference
                best_suit_name = sample_suit.name

    if best_matched_rank_difference < RANK_DIFF_MAX:
        best_matched_rank_name = best_rank_name

    if best_matched_suit_difference < SUIT_DIFF_MAX:
        best_matched_suit_name = best_suit_name

    return best_matched_rank_name, \
           best_matched_suit_name, \
           best_matched_rank_difference, \
           best_matched_suit_difference


def draw_names(image, card):
    x = card.center[0]
    y = card.center[1]
    cv2.circle(
        image,
        (x, y),
        5,
        (255, 0, 0),
        -1
    )

    rank_name = card.best_rank_match
    suit_name = card.best_suit_match

    c = determine_border_color(card)

    cv2.putText(
        image,
        rank_name,
        (x - 60, y - 20),
        font,
        1,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        rank_name,
        (x - 60, y - 20),
        font,
        1,
        c,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        suit_name,
        (x - 60, y + 15),
        font,
        1,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        suit_name,
        (x - 60, y + 15),
        font,
        1,
        c,
        2,
        cv2.LINE_AA
    )

    correctness = str(card.correctness)[0:4]

    cv2.putText(
        image,
        correctness,
        (x - 60, y + 50),
        font,
        1,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        correctness,
        (x - 60, y + 50),
        font,
        1,
        c,
        2,
        cv2.LINE_AA
    )

    return image


def flattener(image, pts, width, height):
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    if width <= 0.8 * height:
        temp_rect[0] = top_left
        temp_rect[1] = top_right
        temp_rect[2] = bottom_right
        temp_rect[3] = bottom_left

    if width >= 1.2 * height:
        temp_rect[0] = bottom_left
        temp_rect[1] = top_left
        temp_rect[2] = top_right
        temp_rect[3] = bottom_right

    if width > 0.8 * height and width < 1.2 * height:
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0]
            temp_rect[1] = pts[0][0]
            temp_rect[2] = pts[3][0]
            temp_rect[3] = pts[2][0]

        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0]
            temp_rect[1] = pts[3][0]
            temp_rect[2] = pts[2][0]
            temp_rect[3] = pts[1][0]

    max_width = 200
    max_height = 300

    return cv2.cvtColor(
        cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(
                temp_rect,
                np.array(
                    [[0, 0], [max_width - 1, 0],
                     [max_width - 1, max_height - 1],
                     [0, max_height - 1]],
                    np.float32
                )
            ),
            (max_width, max_height)
        ),
        cv2.COLOR_BGR2GRAY
    )


def determine_border_color(card):
    v = card.correctness
    if v > 0.5:
        return 0, (((1 - v) * 2) ** 5) * 255, 255
    else:
        return 0, 255, 255 * (v * 2) ** 5
