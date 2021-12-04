import cv2
import numpy as np
from matplotlib import pyplot
import imutils
import easyocr

img = cv2.imread('Images/rear-car-number-plate-500x500.jpg')

# Converting a BGR image to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# We need to filter the image to make it easier for the computer to read it.
img_filter = cv2.bilateralFilter(gray_img, 11, 15, 15)
img_edges = cv2.Canny(img_filter, 30, 200)
contr = cv2.findContours(
    img_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contr = imutils.grab_contours(contr)
contr = sorted(contr, key=cv2.contourArea, reverse=True)[:8]

# Finding quadrangle in the picture
pos = None  # pos as position
for i in contr:
    approx = cv2.approxPolyDP(i, 10, True)

    if(len(approx == 4)):
        pos = approx
        break

# Excluding the license plates from the entire photo
mask = np.zeros(gray_img.shape, np.uint8)
neq_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

# Cropping only the license plates from the entire image
# The colors that are close to white will be stretched out
(x, y) = np.where(mask == 255)  # x and y are coordinates
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

cropping_img = gray_img[x1:x2, y1:y2]

# Reading information
text = easyocr.Reader(['en'])
text = text.readtext(cropping_img)

result = text[0][-2]

final_image = cv2.putText(img, result, (x1 - 200, y2 + 160), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
final_image = cv2.rectangle(img, (y1, x2), (y2, x1), (0, 255, 0), 2)

pyplot.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

pyplot.show()