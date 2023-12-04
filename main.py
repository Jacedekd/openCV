import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np


def predict_food(frame):
    """Predicts the food in a frame and its estimated price."""
    # Resize the frame to 299x299 (expected for InceptionV3)
    img = cv2.resize(frame, (299, 299))

    # Convert frame to image array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the object in the image
    predictions = model.predict(img_array)

    # Decode and get the top prediction with confidence
    decoded_prediction = decode_predictions(predictions, top=1)[0][0]
    predicted_food = decoded_prediction[1]
    confidence = decoded_prediction[2]

    # Print prediction only if confidence is high
    if confidence >= 0.8:
        print(f"Food: {predicted_food}")
        print(f"Estimated Price: {get_price(predicted_food)}")


def get_price(food):
    """Defines the estimated price based on predicted food."""
    if food == "pizza":
        return "$10.00"
    elif food == "burger":
        return "$5.00"
    else:
        return "N/A"


# Initialize camera (if used)
cap = cv2.VideoCapture(0)

while True:
    # Get a frame from the camera
    ret, frame = cap.read()

    # Run prediction if frame is obtained
    if ret:
        predict_food(frame)

# Release resources
cap.release()
cv2.destroyAllWindows()

# Load the pre-trained InceptionV3 model (not shown in original code)
model = InceptionV3(weights="imagenet")

