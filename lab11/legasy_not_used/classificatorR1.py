#Программа нейронной сети прямого распространения для распознавания 6-ти болезней пшеницы 
#по 3D-цифровым описаниям изображений листьев.
#количество цифровых компонентов - 6 (R,G,B,RG,RB,GB)
#количество параметров Харалика - 4. 


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

# Каталог с данными для обучения
train_dir = '../data/train_dir'
# Каталог с данными для тестирования
test_dir = '../data/test_dir'
# Каталог с данными для тестирования
val_dir = '../data/val_dir'
# Размеры изображения
img_width, img_height = 4, 6
# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 100
# Размер мини-выборки
batch_size = 20
# Количество изображений для обучения
nb_train_samples = 100
# Количество изображений для проверки
nb_validation_samples = 30
# Количество изображений для тестирования
nb_test_samples = 20

model = Sequential()

model.add(Dense(20, input_shape=input_shape))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(8))
model.add(Activation('softmax'))
#model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model.summary()

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse')


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=val_generator,
    validation_steps= nb_validation_samples // batch_size,
    epochs=epochs,
    shuffle = True)
model.save('plant_diagnosis_3D.h5')

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Score: ", scores)
