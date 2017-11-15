import os
import wx
import cv2
import numpy
import imutils
from imutils.object_detection import non_max_suppression
import time

def capture_screen(save_file):
    app = wx.App()
    screen = wx.ScreenDC()
    size = screen.GetSize()
    bmp = wx.Bitmap(size[0], size[1])
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
    del mem
    bmp.SaveFile(save_file, wx.BITMAP_TYPE_PNG)

def canny_edge_detect(img_file, edge_file=None):
    img_rgb = cv2.imread(img_file, 0)
    img_gray = cv2.GaussianBlur(img_rgb, (3, 3), 0)
    img_edge = cv2.Canny(img_gray, 50, 150)
    if edge_file is not None:
        cv2.imwrite(edge_file, img_edge)
    return img_edge

def human_detect(img_file, human_file=None):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img = cv2.imread(img_file, 0)
    img = imutils.resize(img, width=min(400, img.shape[1]))
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    if human_file is not None:
        cv2.imwrite(human_file, img)
    return pick

def face_detect(img_file, face_file=None):
    face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath( __file__)), r'opencv\data\haarcascades\haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), r'opencv\data\haarcascades\haarcascade_eye.xml'))

    img_rgb = cv2.imread(img_file)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img_rgb = cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_color = img_rgb[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    if face_file is not None:
        cv2.imwrite(face_file, img_rgb)
    return faces

def object_detect(img_file, obj_file=None):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    COLORS = numpy.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe(os.path.join(os.path.dirname(os.path.realpath(__file__)), r'dnn\Caffe\MobileNetSSD_deploy.prototxt.txt'), os.path.join(os.path.dirname(os.path.realpath(__file__)), r'dnn\Caffe\MobileNetSSD_deploy.caffemodel'))
    image = cv2.imread(img_file)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in numpy.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    if obj_file is not None:
        cv2.imwrite(obj_file, image)
    return detections
