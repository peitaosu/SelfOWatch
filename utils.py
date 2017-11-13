import wx
import cv2

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