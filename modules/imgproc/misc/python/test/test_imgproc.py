#!/usr/bin/env python

from __future__ import print_function

import mlx.core as mx
import cv2 as cv

from tests_common import NewOpenCVTests

class Imgproc_Tests(NewOpenCVTests):

    def test_python_986(self):
        cntls = []
        img = mx.zeros((100,100,3), dtype=mx.uint8)
        color = (0,0,0)
        cnts = mx.array(cntls, dtype=mx.int32).reshape((1, -1, 2))
        try:
            cv.fillPoly(img, cnts, color)
            assert False
        except:
            assert True
