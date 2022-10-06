import keras
import numpy as np
import tensorflow
from PIL import Image
from keras import layers, models, optimizers
import tensorflow_hub as hub
from tensorflow.keras.utils import load_img, img_to_array

def predict_class(img) :
    classifier_model = keras.models.load_model(r'models/final_model.h5', compile = False)
    print('Model Loaded Succafully !!')
    image = Image.open(img)
    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])

    test_image = image.resize((256, 256))
    test_image = tensorflow.keras.utils.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)

    print('Image Loaded and Preprocessed Succafully !!')
    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

if __name__=="__main__":
    image_path = input('Enter Image Path : ')
    print(predict_class(image_path))