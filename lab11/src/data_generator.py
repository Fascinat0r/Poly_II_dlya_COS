# src/data_generator.py

import numpy as np


class DataGenerator:
    def __init__(self, num_samples_per_disease, haralick_params):
        """
        :param num_samples_per_disease: Количество образцов на заболевание.
        :param haralick_params: Словарь с параметрами Харалика.
        """
        self.num_samples_per_disease = num_samples_per_disease
        self.num_diseases = 8
        self.haralick_params = haralick_params

        # Определение цветов и их параметров
        self.color_to_params = self._group_params_by_color(haralick_params)
        self.colors = list(self.color_to_params.keys())
        self.num_features = len(haralick_params)  # Общее число параметров для текущих цветов

        print("Инициализация генератора данных:")
        print(f"- Количество заболеваний: {self.num_diseases}")
        print(f"- Количество образцов на заболевание: {self.num_samples_per_disease}")
        print(f"- Обнаруженные цветовые компоненты: {self.colors}")
        print(f"- Общее количество параметров: {self.num_features}")

    @staticmethod
    def _group_params_by_color(haralick_params):
        """
        Группирует параметры Харалика по цветам.

        :param haralick_params: Словарь параметров Харалика.
        :return: Словарь {цвет: [список параметров]}.
        """
        color_to_params = {}
        for param, values in haralick_params.items():
            color = param[:2] if len(param) > 3 else param[0]
            if color not in color_to_params:
                color_to_params[color] = {}
            color_to_params[color][param] = values
        return color_to_params

    def generate_samples(self, disease_index, num_samples, std_dev=0.032):
        samples = []
        for _ in range(num_samples):
            sample = []
            for color, params in self.color_to_params.items():
                for key, values in params.items():
                    mean_value = values[disease_index]
                    noise = np.random.normal(0, std_dev)
                    value = mean_value + noise
                    sample.append(value)
            # Динамическое определение формы
            num_features = len(sample)
            width = min(6, num_features)  # Максимальная ширина 6
            height = (num_features + width - 1) // width  # Рассчитываем высоту, чтобы вместить все параметры
            sample_array = np.array(
                sample + [0] * (height * width - num_features))  # Заполняем недостающие значения нулями
            sample_array = sample_array.reshape(height, width, 1)
            sample_array = np.repeat(sample_array, 3, axis=-1)  # Преобразуем в 3-канальный RGB
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
        print("\n=== Начало генерации всех наборов данных ===")
        print(f"- Обучающий набор: {num_samples_train} образцов")
        print(f"- Валидирующий набор: {num_samples_val} образцов")
        print(f"- Тестовый набор: {num_samples_test} образцов")

        X_train, y_train = self.generate_dataset(num_samples_train // self.num_diseases)
        print(f"Обучающий набор с {len(X_train)} образцами сгенерирован.")

        X_val, y_val = self.generate_dataset(num_samples_val // self.num_diseases)
        print(f"Валидирующий набор с {len(X_val)} образцами сгенерирован.")

        X_test, y_test = self.generate_dataset(num_samples_test // self.num_diseases)
        print(f"Тестовый набор с {len(X_test)} образцами сгенерирован.")

        print("=== Генерация всех наборов данных завершена ===\n")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
