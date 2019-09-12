import cv2
import numpy as np
import glob
import os
IMG_SIZE=299
path = 'value_train_images/'
alpha=2.2
beta=50  
for i in glob.glob("train_images/*.png"):
    base= os.path.basename(i)
    img = cv2.imread(i)#[:, :, 0]
    new_image = np.zeros(img.shape, img.dtype)
    #img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2RGB)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
    cv2.imwrite(os.path.join(path , base),new_image)
    
