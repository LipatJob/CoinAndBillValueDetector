# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os
from multiprocessing import Pool


def get_encoded_bills(re_encode = False, cuda_available=False):
    input_folder_location = "dataset"
    output_file_location = "model/bills_model.pickle" 

    if not re_encode and os.path.exists(output_file_location):
        print("Encoded dataset found. Reusing")
        return get_encoded(output_file_location)

    print("Encoding dataset")
    return encode_and_save(input_folder_location, output_file_location, use_cnn=cuda_available)


def encode_and_save(input_folder_location, output_file_location, use_cnn=False):
    encoded = encode_directory(input_folder_location, use_cnn)
    pickle_object(encoded, output_file_location)
    return encoded


def encode_directory(folder_location, use_cnn=False):
    imagePaths = list(paths.list_images(folder_location))
    names = [path.split(os.path.sep)[-2] for path in imagePaths]
    images = [cv2.imread(path) for path in imagePaths]

    encodings = []
    if use_cnn:
        for image, path in zip(images, imagePaths):
            print(f"Encoding {path}")
            encodings.append(encode_image_cnn(image))
    else:
        for image, path in zip(images, imagePaths):
            print(f"Encoding {path}")
            encodings.append(encode_image(image))

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
    encoding = face_recognition.face_encodings(rgb, boxes)
    print("Encoding Done")
    return encoding


def encode_image_cnn(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = resize_image(rgb)

    boxes = face_recognition.face_locations(resize, model="cnn")
    encoding = face_recognition.face_encodings(resize, boxes)
    print("Encoding done")
    return encoding 


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


def resize_image(image):
    image = image.copy()

    width = int(image.shape[1])
    height = int(image.shape[0])

    if width > 250 or height > 250:
        factor = 0.5
        width = width * factor
        height = height * factor
        resize_image = cv2.resize(image, (int(height), int(width)))
        return resize_image

    return image


if __name__ == "__main__":
    get_encoded_bills(True, True)
