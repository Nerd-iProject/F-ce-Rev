import tensorflow as tf
model = tf.keras.models.load_model('static/pickle/holly_MobileNet_3(50_class).h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("face_recognizer.tflite", "wb").write(tflite_model)