import pandas as pd


class HaralickDataLoader:
    @staticmethod
    def get_haralick_params(file_path):
        """
        Загружает усредненные параметры Харалика для 8 заболеваний из CSV-файла.

        :param file_path: Путь к CSV-файлу с параметрами Харалика.
        :return: Словарь с параметрами Харалика.
        """
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) != 24:
                raise ValueError(f"Ожидается 24 параметра Харалика, получено {len(df.columns)}.")
            haralick_params = {col: df[col].values for col in df.columns}
            return haralick_params
        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
            return {}
        except Exception as e:
            print(f"Ошибка при загрузке параметров Харалика: {e}")
            return {}


# Пример использования
if __name__ == "__main__":
    loader = HaralickDataLoader()
    haralick_params = loader.get_haralick_params('../data/haralick_parameters.csv')
    print(haralick_params)
