from collections import deque
import numpy as np
import imutils
import cv2

import pyk4a
from pyk4a import Config, PyK4A


orangeLower = (0, 120, 200)
orangeUpper = (18, 255, 255)
pts = deque(maxlen=64)

k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# getters and setters directly get and set on device
k4a.whitebalance = 4500
assert k4a.whitebalance == 4500
k4a.whitebalance = 4510
assert k4a.whitebalance == 4510

i = 0
while 1:
    capture = k4a.get_capture()
    if np.any(capture.color):
        im = capture.color[:, :, :3]

        frame = imutils.resize(im, width=1800)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, orangeLower, orangeUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            if (radius < 300) & (radius > 10 ) : 
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Data_Points.loc[Data_Points.size/3] = [x , y, current_time]
                # print(current_time)

        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            thickness = 2
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


        cv2.imshow('Color', frame)
        key = cv2.waitKey(1)
        if key != -1:
            cv2.destroyAllWindows()
            break
k4a.stop()
    


