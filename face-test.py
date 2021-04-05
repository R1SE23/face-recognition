from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle

# Load face names and encoding from pickle
known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

# Load an image
image = Image.open('test_img.png')

# Detect face(s) and encode them
face_locations = face_recognition.face_locations(np.array(image))
face_encodings = face_recognition.face_encodings(np.array(image), face_locations)

draw = ImageDraw.Draw(image)
face_names = []

# Recognize face(s)
for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    name = known_face_names[best_match_index]

    top, right, bottom, left = face_location
    draw.rectangle([left, top, right, bottom])
    draw.text((left, top), name)
    face_names.append(name)

print(face_names)
image.show()