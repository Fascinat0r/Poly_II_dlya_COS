import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

# Подготовка правильных ответов
# y_train=unils.to_categorical(y_train, 6)

model = tf.keras.models.load_model('plant_diagnosis_3D.h5')

path = '../data/image_for_test/2.png'

image = tf.keras.preprocessing.image.load_img(path, target_size=(4, 6))
x = tf.keras.preprocessing.image.img_to_array(image)
classes = model.predict(np.array([x]))

print(classes)  # Выведет набор вероятностей принадлежности тестового изображения к каждому классу
print('номер класса, предсказанного нейросетью')
print(np.argmax(classes) + 1)
