# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import requests
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
import signal
import sys
import subprocess

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger('crabcam')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

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

def clean_up(signal, frame):
    global STOP
    try:
        STOP = True
        time.sleep(5)
        if conf['show_video']:
            cv2.destroyAllWindows()
        
        task_queue.add_task(create_and_upload, frame_array)
        task_queue.join(timeout=10)
        
        rgb.set_color(OFF)
        camera.close()
    except Exception as e:
        pass
    sys.exit(0)
    

signal.signal(signal.SIGINT, clean_up)
signal.signal(signal.SIGTERM, clean_up)

rgb.set_color(RED)

def create_video(frames):
    prefix = f'/home/pi/Desktop'
    video_path = f'{prefix}/{int(time.time())}-{str(uuid.uuid4())}.avi'

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        conf["fps"],
        (500, 620)
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

    logger.info(f'temp video created at {path}. converting to mp4.')
    mp4_path = f'{path[:-4]}.mp4'
    mp4_filename = mp4_path.split('/')[-1]
    proc = subprocess.Popen(f'ffmpeg -i {path} {mp4_path}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    proc.wait()

    if proc.returncode == 0:
        logger.info(f"successfully converted to mp4 at {mp4_filename}")
        upload_filename = mp4_filename
        # upload_successful = upload_file(mp4_path, 'crabcam', f'videos/{mp4_filename}')
        # slack_data = {'text': f"new file available at: http://crabcam.s3.amazonaws.com/videos/{mp4_filename}", "icon_emoji": ":hermes:"}
    else:
        logger.warn("mp4 conversion failed")
        upload_filename = filename
        # upload_successful = upload_file(path, 'crabcam', f'videos/{filename}')
        # slack_data = {'text': f"new file available at: http://crabcam.s3.amazonaws.com/videos/{filename}", "icon_emoji": ":hermes:"}
    upload_successful = False
    count = 0
    while True:
        upload_successful = upload_file(mp4_path, 'crabcam', f'videos/{upload_filename}')
        if upload_successful:
            slack_data = {'text': f"new file available at: http://crabcam.s3.amazonaws.com/videos/{upload_filename}", "icon_emoji": ":hermes:"}
            break
        
        count += 1
        if count > 50:
            logger.error('giving up on upload after 50 attempts')
            break

        logger.info('retrying upload in 15 seconds')
        time.sleep(15)

    if upload_successful:
        try:
            response = requests.post(conf['slack_webhook'], data=json.dumps(slack_data), headers={'Content-Type': 'application/json'})
        except Exception as e:
            logger.error("failed to post to slack")
        logger.info(f'file uploaded to s3://crabcam/videos/{filename}')
        os.remove(path)
        os.remove(mp4_path)
    else:
        logger.warn('failed to upload to s3')

client = None

task_queue = TaskQueue(1)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.sensor_mode = 4
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]

rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
logger.info("warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

background_object = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=50, detectShadows=True)

frame_array = []

# capture frames from the camera
rgb.set_color(GREEN)
latest_motion_timestamp = time.time()
STOP = False
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    #frame = imutils.rotate(f.array, 180)
    timestamp = datetime.datetime.now()
    text = "IDLE"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(f.array, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        logger.info("starting background model...")
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
    overlay = frame.copy()
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        tfs = np.stack((thresh, thresh, thresh), axis=2)
        ffs = np.stack((frameDelta, frameDelta, frameDelta), axis=2)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) 
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
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
            STOP = True

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    if text != 'IDLE':
        tr = imutils.resize(tfs, height=100)
        fr = imutils.resize(ffs, height=100)

        output_frame = np.zeros((620, 500, 3), dtype="uint8")
        output_frame[:100, :tr.shape[1]] = tr
        output_frame[:100, (500-fr.shape[1]):] = fr
        output_frame[100:, :] = frame

        latest_motion_timestamp = time.time()

        frame_array.append(output_frame)

    if (len(frame_array) > 300 and ((time.time() - latest_motion_timestamp) > 30)) or len(frame_array) > 1200:
        task_queue.add_task(create_and_upload, frame_array)
        latest_motion_timestamp = time.time()
        frame_array = []

    if button.is_pressed():
        logger.info("Pausing ...")
        rgb.set_color(BLUE)
        time.sleep(2)

        while True:
            if button.is_long_press():
                logger.info('stoping ...')
                rgb.set_color(RED)
                STOP = True
                break
            elif button.is_pressed():
                logger.info('Resuming ...')
                rgb.set_color(GREEN)
                background_object = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=50, detectShadows=True)
                break

    if STOP:
        break
