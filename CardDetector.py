import cv2
import numpy as np
import time
import os
import Cards
import Camera

# camera settings
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
tick_frequency = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# creates camera to collect images
camera = Camera.Camera((IMAGE_WIDTH, IMAGE_HEIGHT), FRAME_RATE, 1, 0).start()
# as rpi is laggy it needs time to establish connection to camera
time.sleep(1)

# sample ranks and suits
path = os.path.dirname(os.path.abspath(__file__))
sample_ranks = Cards.load_ranks(path + '/Card_Imgs/')
sample_suits = Cards.load_suits(path + '/Card_Imgs/')


end = 0

# Begin capturing frames
while end == 0:
    time_start = cv2.getTickCount()

    # took an frame
    frame = camera.read()
    frame_processed = Cards.prepare_image(frame)
    # loc cards
    contours_ordered, is_contour_card = Cards.find_cards(frame_processed)

    # anu cards found?
    if len(contours_ordered) != 0:
        identified_cards = []
        idx = 0

        # For each contour detected:
        for i in range(len(contours_ordered)):
            if is_contour_card[i] == 1:
                # cut card from frame
                identified_cards.append(
                    Cards.prepare_card(
                        contours_ordered[i], frame
                    )
                )

                # match a card
                identified_cards[idx].best_rank_match, identified_cards[idx].best_suit_match, identified_cards[idx].rank_difference, identified_cards[idx].suit_difference \
                    = Cards.match_card(identified_cards[idx], sample_ranks, sample_suits)

                # calculate correctnes as difference from minimal needed matched pixels and matched pixels
                identified_cards[idx].correctness = \
                    (Cards.SUIT_DIFF_MAX - identified_cards[idx].suit_difference) / (2 * Cards.SUIT_DIFF_MAX) + \
                    (Cards.RANK_DIFF_MAX - identified_cards[idx].rank_difference) / (2 * Cards.RANK_DIFF_MAX)

                # add result layer to frame
                frame = Cards.draw_names(frame, identified_cards[idx])
                # add extra windows
                try:
                    cv2.imshow("Card " + str(idx) + " warp", identified_cards[idx].warp)
                    cv2.imshow("Card " + str(idx) + " rank", identified_cards[idx].rank_img)
                    cv2.imshow("Card " + str(idx) + " suit", identified_cards[idx].suit_img)
                except TypeError:
                    pass
                idx = idx + 1
        
        # total number of cards on frame
        cv2.putText(frame, "Cards: " + str(idx), (10, 56), font, 0.7, (235, 52, 52), 2, cv2.LINE_AA)

        # draw color correctness depended contours
        if (len(identified_cards) != 0):
            temp_contours = []
            for i in range(len(identified_cards)):
                cv2.drawContours(
                    frame,
                    [identified_cards[i].contour],
                    0,
                    Cards.determine_border_color(
                        identified_cards[i]
                    ),
                    2
                )

    # add framerate
    cv2.putText(frame, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (235, 52, 52), 2, cv2.LINE_AA)

    # Finally, display the image with the identified cards!
    cv2.imshow("Cards", frame)

    # Calculate framerate
    time_end = cv2.getTickCount()
    t = (time_end - time_start) / tick_frequency
    frame_rate_calc = 1 / t
    
    # end program if q pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        end = 1

cv2.destroyAllWindows()
camera.stop()

