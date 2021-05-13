import cv2
import numpy as np
#from sklearn.externals import joblib
import joblib
from skimage.feature import hog
from PIL import Image
cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")


def take_webcame_img():    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            img_name = "img_with_number.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            break
    cam.release()
    cv2.destroyAllWindows()


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            img_name = "img_with_number.png"
            cv2.imwrite(img_name, roi)


def crop(image):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("latest crop selected!")
            cv2.destroyAllWindows()
            break
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        #cv2.waitKey(1)
    # close all open windows
    cv2.destroyAllWindows()

take_webcame_img()
image = cv2.imread('img_with_number.png')
oriImage = image.copy()
crop(image)
image = cv2.imread('img_with_number.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
(thresh, im_binary) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('th2',im_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

pil_image=Image.fromarray(im_binary)
pil_image = pil_image.resize((28,28),resample=4)
#thresh = 200
#fn = lambda x : 255 if x > thresh else 0
#pil_image = pil_image.convert('L').point(fn, mode='1')
pil_image.save('conv_img.bmp')
