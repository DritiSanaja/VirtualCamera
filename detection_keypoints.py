# detection_keypoints.py
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

def detect_face_keypoints(img):
    
    # convert to RGB for MediaPipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = img.shape
    if not results.multi_face_landmarks:
        return []
    landmarks = results.multi_face_landmarks[0].landmark
    # convert normalized coords [0,1] to pixel coords
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    return pts

def draw_keypoints(img, pts):
    # Ensure image is writable and in correct format
    if img.flags['C_CONTIGUOUS'] is False:
        img = np.ascontiguousarray(img)

    # Convert to BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for x, y in pts:
        cv2.circle(img_bgr, (x, y), 1, (0, 255, 0), -1)

    # Convert back to RGB if your pipeline expects RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def replace_face_with_image(frame, pts, replacement_img):
    if not pts:
        return frame

    # Get convex hull of the face keypoints
    hull = cv2.convexHull(np.array(pts))
    x, y, w, h = cv2.boundingRect(hull)

    # Resize replacement image to fit the face region
    replacement_resized = cv2.resize(replacement_img, (w, h))

    # Create mask for the face region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Region of interest in the frame
    roi = frame[y:y+h, x:x+w]

    # Mask for the replacement image
    mask_face = mask[y:y+h, x:x+w]

    # Blend replacement image into the face region
    replacement_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_face))
    replacement_fg = cv2.bitwise_and(replacement_resized, replacement_resized, mask=mask_face)
    dst = cv2.add(replacement_bg, replacement_fg)

    # Place the blended region back into the frame
    frame[y:y+h, x:x+w] = dst
    return frame