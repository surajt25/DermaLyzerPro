import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"skintoneDL\skin_tone_model.tflite")
interpreter.allocate_tensors()

# Input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img_path = r"skintoneDL\download.jpeg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get prediction
predicted_class = np.argmax(output_data)
labels = ["Fair", "Medium", "Dark"]

print("Predicted Class Index:", predicted_class)
print("Predicted Label:", labels[predicted_class])
print("Raw Model Output:", output_data)
