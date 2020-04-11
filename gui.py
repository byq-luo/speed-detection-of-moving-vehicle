#from Tkinter import *
#import tkMessageBox
#import tkfiledialog  #used to open and save dialog object

from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
import os  #communicate with os(file import etc operations)
import time
import uuid  #Universal Unique Identifier,helps in generating random objects of 128 bits as ids. It provides the uniqueness as it generates ids on the basis of time, Computer hardware (MAC etc.).
import math  
from itertools import *        #for working with iterable (sequence-like) data sets.
from PIL import Image,ImageTk    #The ImageTk module contains support to create and modify images


#To use GUI tkinter
root = Tk()
root.geometry('700x700')
root.resizable(width = FALSE ,height= FALSE)


#To dispaly GUI image
Image_open = Image.open("/Users/sudhanshu shekhar/Desktop/Speed Detection 3.7/image/z.png")
image = ImageTk.PhotoImage(Image_open)
logo = Label(root,image=image)   #here image = ImageTk.PhotoImage(Image_open)
logo.place(x=0,y=0)


#To import video 
def callback():
    path = filedialog.askopenfilename()
    e.delete(0, END)  # Remove current text in entry
    e.insert(0, path)  # Insert the 'path'

e = Entry(root,text="")


def upload():
    path = e.get()
    #cap = cv2.VideoCapture(path)

    THRESHOLD_SENSITIVITY = 20 
    BLUR_SIZE = 40 
    BLOB_SIZE = 500 
    BLOB_WIDTH = 150 
    DEFAULT_AVERAGE_WEIGHT = 0.04 
    BLOB_LOCKON_DISTANCE_PX = 80 
    BLOB_TRACK_TIMEOUT = 0.7

    #To change the line thickness and window size
    # Constants for drawing on the frame.
    CIRCLE_SIZE = 0.2
    LINE_THICKNESS = 1
    RESIZE_RATIO = 0.4 # default = 0.4

    #To capture video and analyze
    #input video streaming
    vc = cv2.VideoCapture(path)

    #Find OpenCV version to help the code work efficiently
    #cv2.__version__ gives you the version string
    #The OpenCV version is contained within a special cv2.__version__ variable
    #(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    (major_ver ,  minor, subminor) = cv2.__version__.split('.')
    
    if int(major_ver) < 3 :
            fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)    #to calculate frames per second of the video #frame is one of the many still images which compose the complete moving picture.
            # get vcap property 
            width = vc.get(cv2.CV_CAP_PROP_FRAME_WIDTH)   # float 
            height = vc.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) # float
    else :
            fps = vc.get(cv2.CAP_PROP_FPS)
            # get vcap property 
            width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


    #Kalman-Filter
    def kalman_xy(x, P, measurement, R,
                  motion = np.matrix('0. 0. 0. 0.').T,
                  Q = np.matrix(np.eye(4))):
        """
        Parameters:    
        x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
        P: initial uncertainty convariance matrix
        measurement: observed position
        R: measurement noise 
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        """


        return kalman(x, P, measurement, R, motion, Q,
                      F = np.matrix('''
                          1. 0. 1. 0.;
                          0. 1. 0. 1.;
                          0. 0. 1. 0.;
                          0. 0. 0. 1.
                          '''),
                      H = np.matrix('''
                          1. 0. 0. 0.;
                          0. 1. 0. 0.'''))

    def kalman(x, P, measurement, R, motion, Q, F, H):
        '''
        Parameters:
        x: initial state
        P: initial uncertainty convariance matrix
        measurement: observed position (same shape as H*x)
        R: measurement noise (same shape as H)
        motion: external motion added to state vector x
        Q: motion noise (same shape as P)
        F: next state function: x_prime = F*x
        H: measurement function: position = H*x

        Return: the updated and predicted new values for (x, P)

        See also http://en.wikipedia.org/wiki/Kalman_filter

        This version of kalman can be applied to many different situations by
        appropriately defining F and H 
        '''

        # UPDATE x, P based on measurement m    
        # distance between measured and current position-belief
        y = np.matrix(measurement).T - H * x
        S = H * P * H.T + R  # residual convariance
        K = P * H.T * S.I    # Kalman gain #matrix of gain #The Kalman gain tells you how much I want to change my estimate by given a measurement
        x = x + K*y 
        I = np.matrix(np.eye(F.shape[0])) # identity matrix
        P = (I - K*H)*P  

        # PREDICT x, P based on motion
        x = F*x + motion
        P = F*P*F.T + Q

        return x, P


    def func_kalman_xy(dict_xy):
        x = np.matrix('0. 0. 0. 0.').T
        P = np.matrix(np.eye(4))*1000 # initial uncertainty(doubt)
        observed_x = []
        observed_y = []
        result_array = []

        for item in dict_xy:
            observed_x.append(item[0])
            observed_y.append(item[1])

        N = 20
        result = []
        R = 0.01**2
        
        #zip(): is to map the similar index of multiple containers so that they can be used just using as single entity.
        for meas in zip(observed_x, observed_y):
            x, P = kalman_xy(x, P, meas, R)
            result.append((x[:2]).tolist())  
        kalman_x, kalman_y = zip(*result)

        for i in range(len(kalman_x)):
            pass
            item_a = (round(kalman_x[i][0]), round(kalman_y[i][0])) # round() function which rounds off to the given number of digits
            result_array.append(item_a)
        return result_array


    def calculate_speed (trails, fps):
            # distance: distance on the frame
            # location: x, y coordinates on the frame
            # fps: framerate
            # mmp: meter per pixel
            dist = cv2.norm(trails[0], trails[10])   #Calculates an absolute array norm
            dist_x = trails[0][0] - trails[10][0]
            dist_y = trails[0][1] - trails[10][1]

            mmp_y = 0.2 / (3 * (1 + (3.22 / 432)) * trails[0][1])
            mmp_x = 0.2 / (5 * (1 + (1.5 / 773)) * (width - trails[0][1]))
            real_dist = math.sqrt(dist_x * mmp_x * dist_x * mmp_x + dist_y * mmp_y * dist_y * mmp_y)

            return real_dist * fps * 250 / 3.6


    def get_frame():
            " Grabs a frame from the video vcture and resizes it. "
            rval, frame = vc.read()   
            if rval:
                    (h, w) = frame.shape[:2]
                    frame = cv2.resize(frame, (int(w * RESIZE_RATIO), int(h * RESIZE_RATIO)), interpolation=cv2.INTER_CUBIC)
            return rval, frame


    def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)   ##The tee() function returns several independent iterators (defaults to 2) based on a single original input.
            next(b, None)
            return zip(a, b)

    avg = None
    # A list of "tracked blobs".
    tracked_blobs = []

    a = []
    model_dir = ''
    bgsMOG = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold = 50, detectShadows=0)
    #Background subtraction is a popular method for isolating the moving parts of a scene by segmenting it into background and foreground
    #Creates MOG2 Background Subtractor.
    #history: Length of the history.
    #varThreshold: Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
    #detectShadows:  If true, the algorithm will detect shadows and mark them   

    if vc:
            while True:
                    # Grab the next frame from the camera or video file
                    #The get_frame() method uses the current time in seconds to determine which of the three frames to return at any given moment
                    grabbed, frame = get_frame()

                    if not grabbed:
                            # If we fall into here it's because we ran out of frames
                            # in the video file.
                            break

                    frame_time = time.time()    # time method time() returns the time
                    
                    if grabbed:
                            fgmask = bgsMOG.apply(frame, None, 0.01)     #background subractor
                            # To find the contours of the objects
                            #Contours represent the shapes of objects found in an image.
                            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # cv2.drawContours(frame,contours,-1,(0,255,0),cv2.cv.CV_FILLED,32)
                            try: hierarchy = hierarchy[0]
                            except: hierarchy = []
                            a = []
                            for contour, hier in zip(contours, hierarchy):
                                    (x, y, w, h) = cv2.boundingRect(contour)    #The cv2.boundingRect() function of OpenCV is used to draw an approximate rectangle around the binary image

                                    if w < 80 and h < 80:
                                            continue

                                    center = (int(x + w/2), int(y + h/2))

                                    if center[1] > 320 or center[1] < 150:
                                            continue

                                    # Optionally draw the rectangle around the blob on the frame that we'll show in a UI later
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                    # Look for existing blobs that match this one
                                    closest_blob = None
                                    if tracked_blobs:
                                            # Sort the blobs we have seen in previous frames by pixel distance from this one
                                            closest_blobs = sorted(tracked_blobs, key=lambda b: cv2.norm(b['trail'][0], center))

                                            # Starting from the closest blob, make sure the blob in question is in the expected direction
                                            #blob srores binary data # BLOB (binary large object)
                                            distance = 0.0
                                            distance_five = 0.0
                                            for close_blob in closest_blobs:
                                                    distance = cv2.norm(center, close_blob['trail'][0])
                                                    if len(close_blob['trail']) > 10:
                                                            distance_five = cv2.norm(center, close_blob['trail'][10])
                                                    
                                                    # Check if the distance is close enough to "lock on"
                                                    # BLOB (binary large object)
                                                    if distance < BLOB_LOCKON_DISTANCE_PX:
                                                            # If it's close enough, make sure the blob was moving in the expected direction
                                                            expected_dir = close_blob['dir']
                                                            if expected_dir == 'left' and close_blob['trail'][0][0] < center[0]:
                                                                    continue
                                                            elif expected_dir == 'right' and close_blob['trail'][0][0] > center[0]:
                                                                    continue
                                                            else:
                                                                    closest_blob = close_blob
                                                                    break

                                            if closest_blob:
                                                    
                                                    prev_center = closest_blob['trail'][0]
                                                    if center[0] < prev_center[0]:
                                                            # It's moving left
                                                            closest_blob['dir'] = 'left'
                                                            closest_blob['bumper_x'] = x
                                                    else:
                                                            # It's moving right
                                                            closest_blob['dir'] = 'right'
                                                            closest_blob['bumper_x'] = x + w

                                                    
                                                    closest_blob['trail'].insert(0, center)
                                                    closest_blob['last_seen'] = frame_time
                                                    if len(closest_blob['trail']) > 10:
                                                            closest_blob['speed'].insert(0, calculate_speed (closest_blob['trail'], fps))   #fps is frames per second

                                    if not closest_blob:
                                          
                                            b = dict(
                                                    id=str(uuid.uuid4())[:8],   #  #Universal Unique Identifier(UUID), is a python library which helps in generating random objects of 128 bits as ids
                                                    first_seen=frame_time,
                                                    last_seen=frame_time,
                                                    dir=None,
                                                    bumper_x=None,
                                                    trail=[center],
                                                    speed=[0],
                                                    size=[0, 0],
                                            )
                                            tracked_blobs.append(b)

                            cv2.imshow('BGS', fgmask)   #cv2.imshow() method is used to display an image in a window.

                    if tracked_blobs:
                            # Prune out the blobs that haven't been seen in some amount of time
                            for i in range(len(tracked_blobs) - 1, -1, -1):
                                    if frame_time - tracked_blobs[i]['last_seen'] > BLOB_TRACK_TIMEOUT:
                                            print ("Removing expired track {}".format(tracked_blobs[i]['id']))
                                            del tracked_blobs[i]

                    # Draw information about the blobs on the screen
                    print ('tracked_blobs', tracked_blobs)
                    for blob in tracked_blobs:
                            for (a, b) in pairwise(blob['trail']):
                                    cv2.circle(frame, a, 3, (255, 0, 0), LINE_THICKNESS)  #cv2.circle() method is used to draw a circle on any image.

                                    # print ('blob', blob)
                                    if blob['dir'] == 'left':
                                            pass
                                            cv2.line(frame, a, b, (255, 255, 0), LINE_THICKNESS)
                                    else:
                                            pass
                                            cv2.line(frame, a, b, (0, 255, 255), LINE_THICKNESS)

                                    

                            if blob['speed'] and blob['speed'][0] != 0:

                                    # remove zero elements on the speed list
                                    blob['speed'] = [item for item in blob['speed'] if item != 0.0]
                                    #print ('========= speed list =========', blob['speed'])
                                    ave_speed = np.mean(blob['speed'])
                                    print ('speed', ave_speed)
                                    cv2.putText(frame, str(int(ave_speed)) + 'km/h', (blob['trail'][0][0] - 10, blob['trail'][0][1] + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                                    #cv2.putText() method is used to draw a text string on any image

                    print ('-----------------------------------------')
                    # Show the image from the camera (along with all the lines and annotations)
                    # in a window on the user's screen.
                    cv2.imshow("BGS Method", frame)

                    key = cv2.waitKey(10)  #cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. 
                    if key == 27: # exit on ESC
                            break
     
               
#GUI part
loginbt = Button(root,text = "Select video",width=20,height=2,bg="light blue",fg="black",font="5",relief=RAISED,overrelief=RIDGE,command=callback)
loginbt.place(x =200 ,y=480)
loginbt = Button(root,text = "speed",width=10,height=2,bg="light blue",fg="black",font="5",relief=RAISED,overrelief=RIDGE,command=upload)
loginbt.place(x =400 ,y=480)
root.mainloop()

           
