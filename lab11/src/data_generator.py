# data_generator.py

import numpy as np

class DataGenerator:
    def __init__(self, num_samples_per_disease):
        self.num_samples_per_disease = num_samples_per_disease
        self.num_diseases = 8
        self.num_parameters = 24  # 4 параметра Харалика * 6 цветовых компонентов
        self.haralick_params = self.get_haralick_params()

    def get_haralick_params(self):
        # Усредненные параметры Харалика для 8 заболеваний
        haralick_params = {
            'Rcon': np.array([0.2457, 0.1240, 0.3154, 0.2008, 0.3102, 0.2077, 0.2023, 0.1599]),
            # ... (остальные параметры)
            'GBhom': np.array([0.9400, 0.9687, 0.9618, 0.9583, 0.8518, 0.9591, 0.9484, 0.9482]),
        }
        return haralick_params

    def generate_samples(self, disease_index, num_samples, std_dev=0.032):
        samples = []
        for _ in range(num_samples):
            sample = []
            for key in self.haralick_params:
                mean_value = self.haralick_params[key][disease_index]
                noise = np.random.normal(0, std_dev)
                value = mean_value + noise
                sample.append(value)
            sample_array = np.array(sample).reshape(4, 6)
            # Преобразование в форму (4, 6, 3) для соответствия модели
            sample_array = np.expand_dims(sample_array, axis=-1)
            sample_array = np.repeat(sample_array, 3, axis=-1)
            samples.append(sample_array)
        return samples

    def generate_dataset(self, num_samples_per_disease):
        data = []
        labels = []
        for disease_index in range(self.num_diseases):
            samples = self.generate_samples(disease_index, num_samples_per_disease)
            data.extend(samples)
            labels.extend([disease_index] * num_samples_per_disease)
        return np.array(data), np.array(labels)

    def generate_all_datasets(self, num_samples_train, num_samples_val=30, num_samples_test=20):
        X_train, y_train = self.generate_dataset(num_samples_train // self.num_diseases)
        X_val, y_val = self.generate_dataset(num_samples_val // self.num_diseases)
        X_test, y_test = self.generate_dataset(num_samples_test // self.num_diseases)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
