# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard
import cv2
from filters import equalize_hist_rgb, blur_image, sharpen_image, sobel_edge, linear_transform
from detection import detect_and_replace_face
from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba
from segmentation import segment_background
from detection_keypoints import detect_face_keypoints, draw_keypoints, replace_face_with_image, replace_face_with_image


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator, replacement_imgs, background_img):
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    for sequence in img_source_generator:
        label = 'Original'

        if keyboard.is_pressed('e'):
            sequence = equalize_hist_rgb(sequence)
            label = 'Equalized'
        elif keyboard.is_pressed('z'):  
            sequence = segment_background(sequence, background_img)
            label = 'Background Replaced'
        elif keyboard.is_pressed('b'):
            sequence = blur_image(sequence)
            label = 'Blurred'
        elif keyboard.is_pressed('s'):
            sequence = sharpen_image(sequence)
            label = 'Sharpened'
        elif keyboard.is_pressed('x'):
            sequence = sobel_edge(sequence)
            label = 'Sobel Edge'
        elif keyboard.is_pressed('d'):
            pts = detect_face_keypoints(sequence)
            sequence = replace_face_with_image(sequence, pts, replacement_imgs['d'])
            label = 'Dog Face' 
        elif keyboard.is_pressed('t'):
            sequence = detect_and_replace_face(sequence, replacement_imgs['t'])
            label = 'Trump Face'
        elif keyboard.is_pressed('m'):
            sequence = detect_and_replace_face(sequence, replacement_imgs['m'])
            label = 'Musk Face'
        elif keyboard.is_pressed('l'):  
            sequence = linear_transform(sequence, a=1.2, b=30)  
            label = 'Linear Transform'
        elif keyboard.is_pressed('k'):
            pts = detect_face_keypoints(sequence)
            sequence = draw_keypoints(sequence, pts)
            label = 'Face Mesh Keypoints'
        elif keyboard.is_pressed('f'):  # 'f' for face mesh replacement
            pts = detect_face_keypoints(sequence)
            sequence = replace_face_with_image(sequence, pts, replacement_imgs['m'])
            label = 'Musk Face Mesh'
        elif keyboard.is_pressed('f'): 
            pts = detect_face_keypoints(sequence)
            sequence = replace_face_with_image(sequence, pts, replacement_imgs['m'])
            label = 'Musk Face Mesh'            

        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        sequence = plot_overlay_to_image(sequence, fig)
        sequence = plot_strings_to_image(sequence, [f"Filter: {label}"])
        yield sequence



def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30

    background_img = cv2.imread("sea.jpg")  # Or any image
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

    #face replacment image
    replacement_imgs = {
    'd': cv2.cvtColor(cv2.imread("dog.png"), cv2.COLOR_BGR2RGB),
    't': cv2.cvtColor(cv2.imread("trump.jpg"), cv2.COLOR_BGR2RGB),
    'm': cv2.cvtColor(cv2.imread("musk.jpg"), cv2.COLOR_BGR2RGB)
}

    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True),
            replacement_imgs,
            background_img
        
            
            # or your window screen
            # vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()