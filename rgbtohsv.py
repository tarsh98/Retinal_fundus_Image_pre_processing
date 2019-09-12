import matplotlib.pyplot as plt
import cv2
img = cv2.imread('0bf37ca3156a.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# displaying the image using imshow() function of cv2
# In this : 1st argument is name of the frame
# 2nd argument is the image matrix
fig, ax0 = plt.subplots(figsize=(4, 3))
ax0.imshow(img)
fig.tight_layout()
# converting the colourfull image into HSV format image
# using cv2.COLOR_BGR2HSV argument of
# the cvtColor() function of cv2
# in this :
# ist argument is the image matrix
# 2nd argument is the attribute
HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
fig, ax0 = plt.subplots(figsize=(4, 3))
ax0.imshow(HSV_img)
fig.tight_layout()

img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
fig, ax0 = plt.subplots(figsize=(4, 3))
ax0.imshow(img)
fig.tight_layout()

  
img = cv2.Laplacian(img, cv2.CV_64F)
fig, ax0 = plt.subplots(figsize=(4, 3))
ax0.imshow(img)
fig.tight_layout()

plt.show()

