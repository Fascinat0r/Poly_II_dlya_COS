from itertools import chain, combinations

import pandas as pd


class HaralickDataLoader:
    @staticmethod
    def get_haralick_params(file_path):
        """
        Загружает усредненные параметры Харалика для 8 заболеваний из CSV-файла.

        :param file_path: Путь к CSV-файлу с параметрами Харалика.
        :return: Словарь с параметрами Харалика.
        """
        print(f"Попытка загрузить параметры Харалика из файла: {file_path}")
        try:
            df = pd.read_csv(file_path)
            haralick_params = {col: df[col].values for col in df.columns}
            print("Параметры Харалика успешно загружены.")
            return haralick_params
        except FileNotFoundError:
            print(f"Ошибка: файл {file_path} не найден.")
            return {}
        except Exception as e:
            print(f"Ошибка при загрузке параметров Харалика: {e}")
            return {}

    @staticmethod
    def group_params_by_color(haralick_params):
        """
        Группирует параметры Харалика по цветовому компоненту (например, 'R', 'G', 'B', 'RG', 'RB', 'GB').

        :param haralick_params: Словарь параметров Харалика.
        :return: Словарь {цвет: [список параметров]}.
        """
        color_to_params = {
            'R': ['Rcon', 'Rcor', 'Ren', 'Rhom'],
            'G': ['Gcon', 'Gcor', 'Gen', 'Ghom'],
            'B': ['Bcon', 'Bcor', 'Ben', 'Bhom'],
            'RG': ['RGcon', 'RGcor', 'RGen', 'RGhom'],
            'RB': ['RBcon', 'RBcor', 'RBen', 'RBhom'],
            'GB': ['GBcon', 'GBcor', 'GBen', 'GBhom']
        }

        # Убедимся, что все параметры существуют в haralick_params
        for color, params in color_to_params.items():
            color_to_params[color] = [param for param in params if param in haralick_params]

        return color_to_params

    @staticmethod
    def get_color_combinations(color_to_params):
        """
        Генерирует все возможные комбинации цветов.

        :param color_to_params: Словарь {цвет: [список параметров]}.
        :return: Список комбинаций цветов.
        """
        colors = list(color_to_params.keys())  # ['R', 'G', 'B', 'RG', 'RB', 'GB']
        return list(chain.from_iterable(combinations(colors, r) for r in range(1, len(colors) + 1)))

    @staticmethod
    def get_params_for_combination(color_combination, color_to_params, haralick_params):
        """
        Возвращает параметры Харалика для заданной комбинации цветов.

        :param color_combination: Комбинация цветов (например, ['R', 'G']).
        :param color_to_params: Словарь {цвет: [список параметров]}.
        :param haralick_params: Словарь параметров Харалика.
        :return: Фильтрованный словарь параметров.
        """
        selected_params = {}
        for color in color_combination:
            for param in color_to_params[color]:
                selected_params[param] = haralick_params[param]
        return selected_params
