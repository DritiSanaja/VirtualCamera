import cv2
import numpy as np

def equalize_hist_rgb(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def blur_image(img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(img, kernel_size, 0)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def sobel_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel_img = np.clip(mag, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2RGB)

def linear_transform(img, a=1.0, b=0):
    img = img.astype(float) 
    img = a * img + b
    img = np.clip(img, 0, 255)  
    return img.astype(np.uint8)  
