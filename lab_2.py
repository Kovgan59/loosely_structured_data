import os
import tkinter as tk
import pandas as pd
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()  # Скрываем главное окно tkinter
    file_path = filedialog.askopenfilename()  # Открываем диалог выбора файла
    return file_path

def get_import_ext(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension

def parse_data(file, ext_import):
    match ext_import:
        case '.html':
            df = pd.read_html(file)
        case '.xml':
            df = pd.read_xml(file)
        case '.csv':
            df = pd.read_csv(file)
        case '.json':
            df = pd.read_json(file)
        case _:
            print('Выбрано не верное расширение!')
            print("Расширений для импорта: html, xml, csv, json")
            exit()
    return df

def get_export_ext(df, ext_import):
    print("Введите одно из расширений для экспорта: html, xml, csv, json")
    ext_export = input('Ввод: ')
    if ext_export == ext_import:
        print('Выбрано не верное расширение')
        exit()
    match ext_export:
        case 'html':
            df.to_html('export_html.html', encoding='utf-8', index=False)
        case 'xml':
            df.to_xml('export_xml.xml', encoding='utf-8', index=False)
        case 'csv':
            df.to_csv('export_csv.csv', encoding='utf-8', index=False)
        case 'json':
            df.to_json('export_json.json')

# Использование функции
file = select_file()
ext_import = get_import_ext(file)
print(f"Выбранный файл: {file}")
print(f"Расширение выбранного файла: {ext_import}")

df = parse_data(file, ext_import)
print('============Данные после парсинга============')
print(df)
print('=============================================')

get_export_ext(df, ext_import)

