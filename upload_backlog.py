import glob
import os
import os.path
import time
import logging

from s3_utils import upload_file


logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger('backlog-upload')

for mp4_path in glob.glob('/home/pi/Desktop/15*.mp4'):
    if (time.time() - os.path.getmtime(mp4_path)) / 3600 < 3:
        # ignore if less than 3 hours since create time
        continue

    upload_filename = mp4_path.split('/')[-1]
    avi_path = f"{'/'.join(mp4_path.split('/')[:-1])}/{upload_filename[:-4]}.avi"
       
    while True:
        logger.info(f'uploading {mp4_path} to s3://crabcam/videos/{upload_filename}')
        upload_successful = upload_file(mp4_path, 'crabcam', f'videos/{upload_filename}')
        if upload_successful:
            os.remove(mp4_path)
            if os.path.isfile(avi_path):
                os.remove(avi_path)
            break
        
        count += 1
        if count > 50:
            logger.error('giving up on upload after 50 attempts')
            break

        logger.info('retrying upload in 15 seconds')
        time.sleep(15)
