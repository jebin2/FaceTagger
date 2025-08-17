import dlib
import numpy as np
import cv2


def getFace(img):
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img, 1)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return faces[0]

def encodeFace(image):
  face_location = getFace(image)
  pose_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
  face_landmarks = pose_predictor(image, face_location)
  face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
  face = dlib.get_face_chip(image, face_landmarks)
  encodings = np.array(face_encoder.compute_face_descriptor(face))
  return encodings

def getSimilarity(image1, image2):
  face1_embeddings = encodeFace(image1)
  face2_embeddings = encodeFace(image2)
  return np.linalg.norm(face1_embeddings-face2_embeddings)

img_path_1 = "output/hdbscan/person_000/original_frames/frame_265.6_02656_face08392.jpg"
img_path_2 = "output/hdbscan/person_006/original_frames/frame_281.0_02810_face08478.jpg"

img1 = cv2.imread(img_path_1)
img2 = cv2.imread(img_path_2)

distance = getSimilarity(img1,img2)
print(distance)
if distance < .6:
  print("Faces are of the same person.")
else:
  print("Faces are of different people.")