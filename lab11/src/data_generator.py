# src/data_generator.py

import numpy as np


class DataGenerator:
    def __init__(self, num_samples_per_disease, haralick_params, color_combination, model_type='FFNN'):
        """
        :param num_samples_per_disease: Количество образцов на заболевание.
        :param haralick_params: Словарь с параметрами Харалика.
        :param color_combination: Список цветов, используемых в данных.
        :param model_type: Тип модели ('FFNN' или 'CNN').
        """
        self.num_samples_per_disease = num_samples_per_disease
        self.num_diseases = 8
        self.haralick_params = haralick_params
        self.color_combination = color_combination
        self.model_type = model_type

        # Определение всех параметров
        self.all_colors = ['R', 'G', 'B', 'RG', 'RB', 'GB']
        self.params_per_color = 4  # con, cor, en, hom
        self.total_features = len(self.all_colors) * self.params_per_color  # 24

        print("Инициализация генератора данных:")
        print(f"- Количество заболеваний: {self.num_diseases}")
        print(f"- Количество образцов на заболевание: {self.num_samples_per_disease}")
        print(f"- Обнаруженные цветовые компоненты: {self.color_combination}")
        print(f"- Общее количество параметров: {self.total_features}")
        print(f"- Тип модели: {self.model_type}")

    def generate_samples(self, disease_index, num_samples, std_dev=0.032):
        samples = []
        for _ in range(num_samples):
            sample = []
            for color in self.all_colors:
                if color in self.color_combination:
                    for suffix in ['con', 'cor', 'en', 'hom']:
                        param_name = color + suffix
                        if param_name in self.haralick_params:
                            mean_value = self.haralick_params[param_name][disease_index]
                            noise = np.random.normal(0, std_dev)
                            value = mean_value + noise
                            sample.append(value)
                        else:
                            # Если параметр отсутствует, устанавливаем 0.0
                            sample.append(0.0)
                else:
                    # Для неиспользуемых цветов устанавливаем параметры в 0.0
                    for _ in range(self.params_per_color):
                        sample.append(0.0)
            # Теперь sample имеет 24 элемента

            if self.model_type == 'FFNN':
                # Для FFNN используем плоский вектор
                sample_array = np.array(sample)
            elif self.model_type == 'CNN':
                # Для CNN формируем изображение размером (4, 6, 3)
                height, width = 4, 6
                sample_padded = sample[:height * width]  # Обрезаем, если больше
                sample_padded += [0.0] * (height * width - len(sample_padded))  # Заполняем недостающие
                sample_array = np.array(sample_padded).reshape(height, width, 1)
                sample_array = np.repeat(sample_array, 3, axis=-1)  # Преобразуем в 3-канальный RGB
            else:
                raise ValueError(f"Неизвестный тип модели: {self.model_type}")
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
