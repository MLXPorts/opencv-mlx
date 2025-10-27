#!/usr/bin/env python
'''
    This program demonstrates OpenCV drawing and text output functions by drawing different shapes and text strings
    Usage :
        python3 drawing.py
    Press any button to exit
    '''

# Python 2/3 compatibility
from __future__ import print_function

import mlx.core as mx
import cv2 as cv

# Drawing Lines
def lines():
    for i in range(NUMBER*2):
        pt1, pt2 = [], []
        pt1.append(mx.random.randint(x1, x2))
        pt1.append(mx.random.randint(y1, y2))
        pt2.append(mx.random.randint(x1, x2))
        pt2.append(mx.random.randint(y1, y2))
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        arrowed =  mx.random.randint(0, 6)
        if (arrowed<3):
            cv.line(image, tuple(pt1), tuple(pt2), color, mx.random.randint(1, 10), lineType)
        else:
            cv.arrowedLine(image, tuple(pt1), tuple(pt2), color, mx.random.randint(1, 10), lineType)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY)>=0:
            return

# Drawing Rectangle
def rectangle():
    for i in range(NUMBER*2):
        pt1, pt2 = [], []
        pt1.append(mx.random.randint(x1, x2))
        pt1.append(mx.random.randint(y1, y2))
        pt2.append(mx.random.randint(x1, x2))
        pt2.append(mx.random.randint(y1, y2))
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        thickness = mx.random.randint(-3, 10)
        marker = mx.random.randint(0, 10)
        marker_size = mx.random.randint(30, 80)

        if (marker > 5):
            cv.rectangle(image, tuple(pt1), tuple(pt2), color, max(thickness, -1), lineType)
        else:
            cv.drawMarker(image, tuple(pt1), color, marker, marker_size)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY)>=0:
            return

# Drawing ellipse
def ellipse():
    for i in range(NUMBER*2):
        center = []
        center.append(mx.random.randint(x1, x2))
        center.append(mx.random.randint(y1, y2))
        axes = []
        axes.append(mx.random.randint(0, 200))
        axes.append(mx.random.randint(0, 200))
        angle = mx.random.randint(0, 180)
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        thickness = mx.random.randint(-1, 9)
        cv.ellipse(image, tuple(center), tuple(axes), angle, angle-100, angle + 200, color, thickness, lineType)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY)>=0:
            return

# Drawing Polygonal Curves
def polygonal():
    for i in range(NUMBER):
        pt = [(0, 0)]*6
        pt = mx.resize(pt, (2, 3, 2))
        pt[0][0][0] = mx.random.randint(x1, x2)
        pt[0][0][1] = mx.random.randint(y1, y2)
        pt[0][1][0] = mx.random.randint(x1, x2)
        pt[0][1][1] = mx.random.randint(y1, y2)
        pt[0][2][0] = mx.random.randint(x1, x2)
        pt[0][2][1] = mx.random.randint(y1, y2)
        pt[1][0][0] = mx.random.randint(x1, x2)
        pt[1][0][1] = mx.random.randint(y1, y2)
        pt[1][1][0] = mx.random.randint(x1, x2)
        pt[1][1][1] = mx.random.randint(y1, y2)
        pt[1][2][0] = mx.random.randint(x1, x2)
        pt[1][2][1] = mx.random.randint(y1, y2)
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        alist = []
        for k in pt[0]:
            alist.append(k)
        for k in pt[1]:
            alist.append(k)
        ppt = mx.array(alist)
        cv.polylines(image, [ppt], True, color, thickness = mx.random.randint(1, 10), lineType = lineType)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY) >= 0:
            return

# fills an area bounded by several polygonal contours
def fill():
    for i in range(NUMBER):
        pt = [(0, 0)]*6
        pt = mx.resize(pt, (2, 3, 2))
        pt[0][0][0] = mx.random.randint(x1, x2)
        pt[0][0][1] = mx.random.randint(y1, y2)
        pt[0][1][0] = mx.random.randint(x1, x2)
        pt[0][1][1] = mx.random.randint(y1, y2)
        pt[0][2][0] = mx.random.randint(x1, x2)
        pt[0][2][1] = mx.random.randint(y1, y2)
        pt[1][0][0] = mx.random.randint(x1, x2)
        pt[1][0][1] = mx.random.randint(y1, y2)
        pt[1][1][0] = mx.random.randint(x1, x2)
        pt[1][1][1] = mx.random.randint(y1, y2)
        pt[1][2][0] = mx.random.randint(x1, x2)
        pt[1][2][1] = mx.random.randint(y1, y2)
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        alist = []
        for k in pt[0]:
            alist.append(k)
        for k in pt[1]:
            alist.append(k)
        ppt = mx.array(alist)
        cv.fillPoly(image, [ppt], color, lineType)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY) >= 0:
            return

# Drawing Circles
def circles():
    for i in range(NUMBER):
        center = []
        center.append(mx.random.randint(x1, x2))
        center.append(mx.random.randint(y1, y2))
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        cv.circle(image, tuple(center), mx.random.randint(0, 300), color, mx.random.randint(-1, 9), lineType)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY) >= 0:
            return

# Draws a text string
def string():
    for i in range(NUMBER):
        org = []
        org.append(mx.random.randint(x1, x2))
        org.append(mx.random.randint(y1, y2))
        color = "%06x" % mx.random.randint(0, 0xFFFFFF)
        color = tuple(int(color[i:i+2], 16) for i in (0, 2 ,4))
        cv.putText(image, "Testing text rendering", tuple(org), mx.random.randint(0, 8), mx.random.randint(0, 100)*0.05+0.1, color, mx.random.randint(1, 10), lineType)
        cv.imshow(wndname, image)
        if cv.waitKey(DELAY) >= 0:
            return


def string1():
    textsize = cv.getTextSize("OpenCV forever!", cv.FONT_HERSHEY_COMPLEX, 3, 5)
    org = (int((width - textsize[0][0])/2), int((height - textsize[0][1])/2))
    for i in range(0, 255, 2):
        image2 = mx.array(image) - i
        cv.putText(image2, "OpenCV forever!", org, cv.FONT_HERSHEY_COMPLEX, 3, (i, i, 255), 5, lineType)
        cv.imshow(wndname, image2)
        if cv.waitKey(DELAY) >= 0:
            return

if __name__ == '__main__':
    print(__doc__)
    wndname = "Drawing Demo"
    NUMBER = 100
    DELAY = 5
    width, height = 1000, 700
    lineType = cv.LINE_AA  # change it to LINE_8 to see non-antialiased graphics
    x1, x2, y1, y2 = -width/2, width*3/2, -height/2, height*3/2
    image = mx.zeros((height, width, 3), dtype = mx.uint8)
    cv.imshow(wndname, image)
    cv.waitKey(DELAY)
    lines()
    rectangle()
    ellipse()
    polygonal()
    fill()
    circles()
    string()
    string1()
    cv.waitKey(0)
    cv.destroyAllWindows()