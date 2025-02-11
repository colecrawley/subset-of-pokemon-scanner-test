import os 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

categories = os.listdir(r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\train")
categories.sort()
print(categories)

#load the saved model

path_for_saved_model = r"C:\Users\lf220\Desktop\mobilenet project\dataset_for_model\cardV2.h5"
model = tf.keras.models.load_model(path_for_saved_model)

print(model.summary())

def classfiy_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((224,224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)


    x = preprocess_input(x)

    print(x.shape)

    pred = model.predict(x)
    categoryValue = np.argmax(pred , axis = 1)

    categoryValue = categoryValue[0]
    print(categoryValue)

    result = categories[categoryValue]

    return result

imagePath = r"C:\Users\lf220\Desktop\mobilenet project\test_images\squirtle.jpg"
resultText = classfiy_image(imagePath)
print(resultText)

img = cv2.imread(imagePath)
img = cv2.putText(img, resultText, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()