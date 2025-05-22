import tensorflow as tf
import cv2
import numpy as np
import os

class FramePredictor:
    def __init__(self, model_path, frames_folder):
        self.model = tf.keras.models.load_model(model_path)
        self.frames_folder = frames_folder
        self.classnames = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to match model input size
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)  # Expand dimensions to (1, 48, 48, 3)

    def predict_frames(self):
        preds = []
        frame_files = sorted(os.listdir(self.frames_folder))
        
        for frame in frame_files:
            if frame.endswith('.jpg'):
                img_path = os.path.join(self.frames_folder, frame)
                img = self.preprocess_image(img_path)
                prediction = self.model.predict(img)[0]  # Get prediction for each frame
                class_pred = self.classnames[np.argmax(prediction)]
                preds.append(class_pred)
                print(f'{frame}: {class_pred}')
        
        return preds
    
    

# Example usage
# predictor = FramePredictor('path_to_model.h5')
# predictions = predictor.predict_frames()
# print('Predictions:', predictions)
