from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import random
import yaml
import os

def create_and_save_yaml(file_name: str):
    n = int(input("Введите количество значений: "))
    time = list(range(1, n + 1)) # Создаем даты от 1 до n
    values = [random.random() for _ in range(n)] # Создаем n случайных значений

    data = {'time': time, 'values': values}

    # Получаем текущую рабочую директорию
    current_directory = os.getcwd()
    # Создаем полный путь к файлу, добавляя имя файла к текущей директории
    file_path = os.path.join(current_directory, file_name)

    with open(file_path, 'w') as file:
        yaml.dump(data, file)

    print(f"Файл {file_path} успешно создан.")

@dataclass
class TimeValueData:
    time: List[str]
    values: List[float]

def parse_yaml_file(file_path: str) -> TimeValueData:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return TimeValueData(time=data['time'], values=data['values'])

def plot_time_value(data: TimeValueData):
    plt.plot(data.time, data.values)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Value Data')
    plt.show(block=True)

if __name__ == "__main__":
    file_name = 'file.yaml'
    create_and_save_yaml(file_name)
    data = parse_yaml_file(file_name)
    plot_time_value(data)