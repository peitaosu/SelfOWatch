import wx
import cv2
import numpy
import imutils
from imutils.object_detection import non_max_suppression

def capture_screen(save_file):
    wx.App()
    screen = wx.ScreenDC()
    size = screen.GetSize()
    bmp = wx.EmptyBitmap(size[0], size[1])
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
    del mem
    bmp.SaveFile(save_file, wx.BITMAP_TYPE_PNG)

def canny_edge_detect(img_file, edge_file):
    img_rgb = cv2.imread(img_file, 0)
    img_gray = cv2.GaussianBlur(img_rgb,(3,3),0)
    img_edge = cv2.Canny(img_gray, 50, 150)
    cv2.imwrite(edge_file, img_edge)

def human_detect(img_file, human_file):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    img = cv2.imread(img_file, 0)
    img = imutils.resize(img, width=min(400, img.shape[1]))
    (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imwrite(human_file, img)
