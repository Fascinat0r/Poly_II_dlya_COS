import os
import shutil


def clear_data(results_file, stats_dir, plots_dir):
    """
    Очистка результатов, прогонов и графиков.
    """
    if os.path.exists(results_file):
        os.remove(results_file)
    if os.path.exists(stats_dir):
        shutil.rmtree(stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    print("Все данные очищены.")
