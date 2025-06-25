import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_replace_face(img, replacement_img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        replacement_resized = cv2.resize(replacement_img, (w, h))
        img[y:y+h, x:x+w] = replacement_resized
    return img
