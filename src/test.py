import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('/home/auerx/hm/8-semester/zdo/incisions/images/default/SA_20230222-081220_incision_crop_0.jpg')
row, col = im.shape[:2]
bottom = im[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

border_size = 10
border = cv2.copyMakeBorder(
    im,
    top=border_size,
    bottom=border_size,
    left=border_size,
    right=border_size,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean, mean, mean]
)


plt.imshow(border)
plt.show()
test=0