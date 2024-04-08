import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import SpectralClustering

from scipy.spatial.distance import pdist, squareform

'''
Только при первом запуске
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
'''

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def select_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    print(directory)
    return directory

def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def preprocess_text(text):
    tokens = word_tokenize(text) # получение токенов
    tokens = [token.lower() for token in tokens if token.isalpha()] # отчистка символов (цифры, знаки) + нижний регистр
    tokens = [token for token in tokens if token not in stopwords.words('russian')] # исключение стоп-слов (без смысловой нагрузки)
    tokens = [lemmatizer.lemmatize(token) for token in tokens] # лемматизация - базовая форма слов
    return tokens

def load_and_process_documents(directory):
    documents = []
    all_files = list_all_files(directory)
    for file in all_files:
        if file.endswith('.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                processed_text = preprocess_text(text)
                documents.append(processed_text)
    return documents

lemmatizer = WordNetLemmatizer()
dir_path = select_directory()
documents = load_and_process_documents(dir_path)
corpus = [' '.join(doc) for doc in documents]

# Векторизация
vectorizer = CountVectorizer(max_features=150, ngram_range=(1, 1))
X = vectorizer.fit_transform(corpus)

# Матрица растояний
dist_matrix = pdist(X.toarray(), metric='cosine')
dist_matrix = squareform(dist_matrix)

# Кластеризация
delta = 4.0
similarity_matrix = np.exp(- dist_matrix ** 2 / (2. * delta ** 2))
clustering = SpectralClustering(affinity='precomputed', n_clusters=3, assign_labels='discretize', random_state=0)
clusters = clustering.fit_predict(similarity_matrix)

# Вывод графиков
x = dist_matrix[:, 0]
y = dist_matrix[:, 1]

plt.figure(figsize=(10, 10))
plt.title('Матрица расстояний')
sns.scatterplot(x=x, y=y, hue=dist_matrix[:, 1], palette='coolwarm', marker='o')
plt.show()

X_dense = X.toarray()

unique_clusters = len(set(clusters))
colors = sns.color_palette("hsv", unique_clusters)

plt.figure(figsize=(10, 10))
plt.title('Кластеры')
sns.scatterplot(x=X_dense[:, 0], y=X_dense[:, 1], hue=clusters, palette=colors)
plt.show()