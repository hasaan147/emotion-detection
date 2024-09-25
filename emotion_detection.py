from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('emotion_detection_model.keras')

# Emotion labels based on your training data
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        # Convert image to grayscale
        image_gray = image.convert('L')  # Convert to grayscale

        # Resize to 48x48 pixels
        image_resized = image_gray.resize((48, 48))

        # Normalize the image (0-255 to 0-1)
        image_normalized = np.array(image_resized) / 255.0

        # Reshape to match the model input shape
        image_reshaped = np.reshape(image_normalized, (1, 48, 48, 1))

        # Predict the emotion
        prediction = model.predict(image_reshaped)
        emotion_index = np.argmax(prediction[0])
        emotion = emotion_labels[emotion_index]

        # Put emotion text on the frame
        cv2.putText(img, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
