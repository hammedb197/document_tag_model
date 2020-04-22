import cv2
import pytesseract
import imutils
import numpy as np
import textract
from collections import Counter

def extract_from_images(img):
  # img = cv2.imread(file)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  (_, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   # convert2binary
  contours = cv2.findContours(~binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if imutils.is_cv4() else contours[1]
  heights = [cv2.boundingRect(contour)[3] for contour in contours]
  average_ = sum(heights)/len(heights)

  mask = np.ones(img.shape[:2], dtype="uint8") * 255
  #create empty image of the size of the image
  for c in contours:
      [x, y, w, h] = cv2.boundingRect(c)
      if h > average_ * 2:
          cv2.drawContours(mask, [c], -1, 0, -1)
          
  title = pytesseract.image_to_string(mask)
  content = pytesseract.image_to_string(img)
  if len(content) == 0:
      content = textract.process(content)
  return content
