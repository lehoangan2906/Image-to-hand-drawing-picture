import cv2 as cv
import numpy as np
import random 
from edges import Edges
from background import Background
from PIL import Image, ImageOps

filename = "photo.jpg"

class Sketch:
    def __init__ (self, image):
        self.img = cv.imread(image)

    def sketch(self):
        grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        inv = 255 - grey
        blur = cv.GaussianBlur(inv, (13, 13), 0)
        return cv.divide(grey, 255-blur, scale = 256)

img = Image.open(filename)

sketch = Sketch(filename).sketch()
cv.imwrite("sketch.png", sketch)    # is used to save an image to any storage device
bg = Background(img.size, octaves = 6).background()
edges = Edges(filename).edges()

mask = edges[3]
sketch = cv.bitwise_and(sketch, edges, edges)
(thresh, sketch) = cv.threshold(sketch, 240, 255, cv.THRESH_BINARY)

h, w = sketch.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
sketchColor = cv.cvtColor(sketch, cv.COLOR_GRAY2RGBA)
white = np.all(sketchColor == [255, 255, 255, 255], axis = -1)
sketchColor[white, -1] = 0
cv.imwrite("final.png", sketchColor)
final = Image.fromarray(sketchColor)
final.show()
