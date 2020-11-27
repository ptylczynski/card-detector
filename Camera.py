from threading import Thread
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera


class Camera:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.pi_camera = PiCamera()
        self.pi_camera.resolution = resolution
        self.pi_camera.framerate = framerate
        self.raw_capture = PiRGBArray(self.pi_camera, size=resolution)
        self.frame_stream =\
            self.pi_camera.capture_continuous(
                self.raw_capture,
                format="bgr",
                use_video_port=True
            )
        self.frames = []
        self.is_working = True

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        for f in self.frame_stream:
            self.frames = f.array
            self.raw_capture.truncate(0)
            if not self.is_working:
                self.frame_stream.close()
                self.raw_capture.close()
                self.pi_camera.close()

    def read(self):
        return self.frames

    def stop(self):
        self.is_working = False
