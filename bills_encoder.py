# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os
from multiprocessing import Pool


def get_encoded_bills(re_encode = False):
    input_folder_location = "dataset"
    output_file_location = "model/bills_model.pickle" 

    if not re_encode and os.path.exists(output_file_location):
        return get_encoded(output_file_location)

    return encode_and_save(input_folder_location, output_file_location, use_cnn=False)


def encode_and_save(input_folder_location, output_file_location, use_cnn=True):
    encoded = encode_directory(input_folder_location, use_cnn)
    pickle_object(encoded, output_file_location)
    return encoded


def encode_directory(folder_location, use_cnn=False):
    imagePaths = list(paths.list_images(folder_location))
    names = [path.split(os.path.sep)[-2] for path in imagePaths]
    images = [cv2.imread(path) for path in imagePaths]

    encodings = []
    if use_cnn:
        encodings = map(encode_image_cnn, images)
    else:
        with Pool(4) as pool:
            encodings = pool.map(encode_image, images)

    known_names = []
    known_encodings = []

    for name, sub_encoding in zip(names, encodings):
        for encoding in sub_encoding:
            known_names.append(name)
            known_encodings.append(encoding)

    return {"encodings": known_encodings, "names": known_names}


def flatten(t):
    return [item for sublist in t for item in sublist]


def encode_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    return face_recognition.face_encodings(rgb, boxes)


def encode_image_cnn(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    return face_recognition.face_encodings(rgb, boxes)


def pickle_object(data, file_location):
    with open(file_location, "wb") as f:
        f.write(pickle.dumps(data))


def get_encoded(file_location):
    if not os.path.isfile(file_location):
        return None

    with open(file_location, "rb") as f:
        file_content = f.read()

    # Add additional validation if needed

    return pickle.loads(file_content)