# src/data_loader.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    def __init__(self, train_dir, val_dir, test_dir, img_width=28, img_height=28, batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1. / 255)

    def get_train_generator(self):
        return self.datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='sparse'
        )

    def get_val_generator(self):
        return self.datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='sparse'
        )

    def get_test_generator(self):
        return self.datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='sparse',
            shuffle=False
        )
