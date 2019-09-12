import matplotlib.pyplot as plt
import cv2
import  os
import glob
import numpy as np
from skimage import data
from skimage.color import rgb2hsv

frame=cv2.imread("0bf37ca3156a.png")
hsv = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)





##import matplotlib.pyplot as plt
##import cv2
##import  os
##import glob
##from skimage import data
##from skimage.color import rgb2hsv
##IMG_SIZE=299
##path="value_train_images/"
##for i  in glob.glob("train_images/*.png"):
##    base=os.path.basename(i)
##    rgb_img = cv2.imread(i)
##    rgb_img = cv2.cvtColor(rgb_img,cv2.cv2.COLOR_BGR2RGB)
##    image = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
##    hsv_img = rgb2hsv(rgb_img)
##    hue_img = hsv_img[:, :, 0]
##    value_img = hsv_img[:, :, 2]
##    
##    cv2.imwrite(os.path.join(path , base),image)
####import numpy as np 
##import pandas as pd 
##import os
##import cv2 
##import PIL 
##import gc
##import psutil
##import matplotlib.pyplot as plt
##from tqdm import tqdm
##from math import ceil
##import glob
##
##
##class HomomorphicFilter:
##
##    def __init__(self, a = 0.5, b = 1.5):
##        self.a = float(a)
##        self.b = float(b)
##
##    # Filters
##    def __butterworth_filter(self, I_shape, filter_params):
##        P = I_shape[0]/2
##        Q = I_shape[1]/2
##        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
##        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
##        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
##        return (1 - H)
##
##    def __gaussian_filter(self, I_shape, filter_params):
##        P = I_shape[0]/2
##        Q = I_shape[1]/2
##        H = np.zeros(I_shape)
##        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
##        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
##        H = np.exp((-Duv/(2*(filter_params[0])**2)))
##        return (1 - H)
##
##    # Methods
##    def __apply_filter(self, I, H):
##        H = np.fft.fftshift(H)
##        I_filtered = (self.a + self.b*H)*I
##        return I_filtered
##
##    def filter(self, I, filter_params, filter='gaussian', H = None):
##
##        #  Validating image
##        if len(I.shape) is not 2:
##            raise Exception('Improper image')
##
##        # Take the image to log domain and then to frequency domain 
##        I_log = np.log1p(np.array(I, dtype="float"))
##        I_fft = np.fft.fft2(I_log)
##
##        # Filters
##        if filter=='butterworth':
##            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
##        elif filter=='gaussian':
##            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
##        elif filter=='external':
##            print('external')
##            if len(H.shape) is not 2:
##                raise Exception('Invalid external filter')
##        else:
##            raise Exception('Selected filter not implemented')
##        
##        # Apply filter on frequency domain then take the image back to spatial domain
##        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
##        I_filt = np.fft.ifft2(I_fft_filt)
##        I = np.exp(np.real(I_filt))-1
##        return np.uint8(I)
##
##
##if __name__ == "__main__":
##    IMG_SIZE=299
##    path = 'value_train_images/'
##    for i in glob.glob("train_images/*.png"):
##        base= os.path.basename(i)
##    ##    img = cv2.imread(i)
##    ##    img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
##    ##    img = crop_image_from_gray(img)
##    ##    img = cv2.resize(img,(IMG_DIM,IMG_DIM))
##    ##    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), IMG_DIM/10),-4, 128)
##    ##    cv2.imwrite(os.path.join(path , base),img)
##        img = cv2.imread(i)#[:, :, 0]
##        img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2RGB)
##        img = cv2.cvtColor(img,cv2.cv2.COLOR_RGB2GRAY)
##        homo_filter = HomomorphicFilter(a = 0.90, b = 1.45)
##        img_filtered = homo_filter.filter(I=img, filter_params=[20,3])
##        image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
##        #edges = cv2.Canny(img_filtered,100,200)
##        cv2.imwrite(os.path.join(path , base),img)
##    
##    
