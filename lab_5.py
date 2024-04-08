import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

'''
Только при первом запуске
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
'''

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pymorphy3

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
    morph = pymorphy3.MorphAnalyzer()    
    tokens = [morph.parse(token)[0].normal_form for token in tokens if token.isalnum() and not token.isdigit()]
    tokens = [token for token in tokens if token not in stopwords.words('russian')]
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

def plot_top_words(model, feature_names, n_top_words, title):
    n_components = model.n_components
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10)) # Создаем сетку из двух строк и пяти столбцов
    axes = axes.flatten() # Преобразуем сетку в одномерный массив для удобства индексации
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 12})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=20)

    # Удаляем пустые оси, если количество компонентов меньше 10
    if n_components < 10:
        for i in range(n_components, 10):
            fig.delaxes(axes[i])

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

n_features = 1000
n_components = 10
n_top_words = 20

if __name__ == '__main__':

    dir_path = select_directory()
    documents = load_and_process_documents(dir_path)
    corpus = [' '.join(doc) for doc in documents]

    countV = CountVectorizer(max_features=1000)                                        
    X_countV = countV.fit_transform(corpus)

    tfidfV = TfidfVectorizer(max_features=1000)                                        
    X_tfidfV = tfidfV.fit_transform(corpus)
    
    feature_names = tfidfV.get_feature_names_out() 
    
    # Применяем модель NMF
    nmf = NMF(n_components=n_components, random_state=1, l1_ratio=.5).fit(X_countV)
    plot_top_words(nmf, feature_names, n_top_words, 'Topics in NMF model (Frobenius norm)')

    # Применяем модель NMF с другими параметрами
    nmf_k = NMF(n_components=n_components, random_state=1,
            beta_loss='kullback-leibler', solver='mu', max_iter=1000, l1_ratio=.5).fit(X_tfidfV)
    plot_top_words(nmf_k, feature_names, n_top_words, 'Topics in NMF model (generalized Kullback-Leibler divergence)')
    
    # Применяем модель LDA
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(X_tfidfV)
    plot_top_words(lda, feature_names, n_top_words, 'Topics in LDA model')