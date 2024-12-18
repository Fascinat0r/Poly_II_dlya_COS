import os

import pandas as pd


def process_and_round_csv(input_file, output_file):
    """
    Открывает CSV-файл, удаляет первый столбец, округляет числа с плавающей точкой до 6 знаков
    и сохраняет результат в новый файл.

    Args:
        input_file (str): Путь к входному CSV-файлу.
        output_file (str): Путь к выходному CSV-файлу.
    """
    # Чтение исходного CSV файла
    df = pd.read_csv(input_file)

    # Удаление первого столбца
    df = df.iloc[:, 1:]

    # Округление всех значений с плавающей точкой до 6 знаков после запятой
    df = df.applymap(lambda x: round(x, 6) if isinstance(x, float) else x)

    # Сохранение обработанного DataFrame в новый CSV файл
    df.to_csv(output_file, index=False)
