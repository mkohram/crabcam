# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
# import dropbox
import imutils
import json
import time
import cv2
import os
import uuid
import numpy as np
from task_queue import TaskQueue
from s3_utils import upload_file
import logging
from squid import *
from button import *
import RPi.GPIO as GPIO

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger('crabcam')

class LongPressButton(Button):
    def __init__(self, button_pin, debounce=0.05, long_press=5):
        super().__init__(button_pin, debounce)

        self.long_press = long_press

    def is_long_press(self):
        now = time.time()
        if GPIO.input(self.BUTTON_PIN) == False:
            time.sleep(self.DEBOUNCE)
            # wait for button release
            press = time.time()
            while not GPIO.input(self.BUTTON_PIN):
                pass

            if (time.time() - press) >= self.long_press:
                return True
        return False

rgb = Squid(18, 23, 24)
button = LongPressButton(25, debounce=0.1, long_press=5)
rgb.set_color(RED)

def create_video(frames):
    prefix = f'/home/pi/Desktop'
    video_path = f'{prefix}/{int(time.time())}-{str(uuid.uuid4())}.avi'

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        10.0,
        (500, 381)
    )

    for i in range(len(frames)):
        # writing to a image array
        out.write(frames[i])

    out.release()

    return video_path

def create_and_upload(frames):
    if len(frames) < 50:
        logger.info('Video too short. Not saving.')
        return

    path = create_video(frames)
    filename = path.split('/')[-1]
    logger.info(f'temp video created at {path}')
    upload_successful = upload_file(path, 'crabcam', f'videos/{filename}')

    if upload_successful:
        logger.info(f'file uploaded to s3://crabcam/videos/{filename}')
        os.remove(path)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

task_queue = TaskQueue(1)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

background_object = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=50, detectShadows=True)

frame_array = []
# out = cv2.VideoWriter('/home/pi/Desktop/vid.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10.0, (500, 475))

# capture frames from the camera
rgb.set_color(GREEN)
first_frame_timestamp = time.time()
stop = False
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame = imutils.rotate(f.array, 180)
    timestamp = datetime.datetime.now()
    text = "IDLE"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue
    #
    # # accumulate the weighted average between the current frame and
    # # previous frames, then compute the difference between the current
    # # frame and running average
    # cv2.accumulateWeighted(gray, avg, 0.5)
    # frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    frameDelta = background_object.apply(gray)

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        tfs = np.stack((thresh, thresh, thresh), axis=2)
        ffs = np.stack((frameDelta, frameDelta, frameDelta), axis=2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(tfs, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(ffs, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion"

    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Motion status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the security feed
        cv2.imshow("RGB feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            stop = True

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    if text != 'IDLE':
        tr = imutils.resize(tfs, height=100)
        fr = imutils.resize(ffs, height=100)

        output_frame = np.zeros((381, 500, 3), dtype="uint8")
        output_frame[:100, :tr.shape[1]] = tr
        output_frame[:100, (500-fr.shape[1]):] = fr
        # output_frame[:100, 300:] = fr
        output_frame[100:, :] = frame

        if len(frame_array) == 0:
            first_frame_timestamp = time.time()

        frame_array.append(output_frame)

    if len(frame_array) > 300 or ((time.time() - first_frame_timestamp) > 2 * 3600):
        task_queue.add_task(create_and_upload, frame_array)
        frame_array = []

    if button.is_pressed():
        logger.info("Pausing ...")
        rgb.set_color(BLUE)
        time.sleep(2)

        while True:
            if button.is_long_press():
                logger.info('stoping ...')
                rgb.set_color(RED)
                stop = True
                break
            elif button.is_pressed():
                logger.info('Resuming ...')
                rgb.set_color(GREEN)
                break

    if stop:
        break

if conf['show_video']:
    cv2.destroyAllWindows()

if len(frame_array) > 100:
    task_queue.add_task(create_and_upload, frame_array)
task_queue.join()

rgb.set_color(OFF)

