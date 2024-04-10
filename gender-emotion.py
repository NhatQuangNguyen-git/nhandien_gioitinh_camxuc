import cv2
from keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv

# Load emotion detection model
json_file = open("emotiondetector4.json", "r")
model_json = json_file.read()

json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotiondetector4.h5")
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load gender detection model
gender_model = load_model('gender_detection.model')
gender_classes = ['man', 'woman']

# Open webcam
webcam = cv2.VideoCapture(0)

# Load face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    # Apply face detection
    faces, confidences = cv.detect_face(frame)

    # Loop through detected faces
    for face, confidence in zip(faces, confidences):
        # Get corner points of face rectangle
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess for gender detection model
        face_crop_gender = cv2.resize(face_crop, (96, 96))
        face_crop_gender = face_crop_gender.astype("float") / 255.0
        face_crop_gender = img_to_array(face_crop_gender)
        face_crop_gender = np.expand_dims(face_crop_gender, axis=0)

        # Apply gender detection on face
        gender_conf = gender_model.predict(face_crop_gender)[0]
        gender_idx = np.argmax(gender_conf)
        gender_label = gender_classes[gender_idx]
        gender_label = "{}: {:.2f}%".format(gender_label, gender_conf[gender_idx] * 100)

        # Preprocess for emotion detection model
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if face_gray.shape[0] >= 48 and face_gray.shape[1] >= 48:
            face_gray = cv2.resize(face_gray, (48, 48))
            face_gray = face_gray.reshape(1, 48, 48, 1)
            face_gray = face_gray / 255.0
        else:
            continue  # Bỏ qua khuôn mặt không đủ kích thước

        # Apply emotion detection on face
        emotion_pred = emotion_model.predict(face_gray)
        emotion_idx = np.argmax(emotion_pred)
        emotion_label = emotion_labels[emotion_idx]

        # Write gender and emotion labels above face rectangle
        cv2.putText(frame, gender_label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (startX, startY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Gender and Emotion Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()