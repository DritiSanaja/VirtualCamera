import mediapipe as mp
import numpy as np
import cv2

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # because model 1 is more accurat

def segment_background(frame, background_img=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #opencv uses BGR thats why we need to turn into RGB
    results = segmentor.process(rgb) # the segmentation process wher close to 1 belong to person otherwiser background

    condition = results.segmentation_mask > 0.6

    if background_img is None:
        blurred = cv2.GaussianBlur(frame, (55, 55), 0)
        output = np.where(condition[..., None], frame, blurred)
    else:
        background_img = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))
        output = np.where(condition[..., None], frame, background_img)

    return output
