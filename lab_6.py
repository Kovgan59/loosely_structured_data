import pyaudio
import numpy as np
import whisper
import subprocess
import wave
import webbrowser

# '''
# Только при первом запуске
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# '''

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pymorphy3

# Функция для записи аудио
def record_audio(duration=3, rate=44100, channels=1, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []
    print("Начинается запись речи...") # Вывод о начале записи
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.int16))
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Запись речи завершена.") # Вывод о завершении записи
    # Сохранение аудио в формате WAV
    audio_data = b''.join(frames)
    save_audio_as_wav(audio_data, 'output.wav')

    return np.concatenate(frames)

def save_audio_as_wav(audio_data, filename):
    # Параметры аудио
    nchannels = 1 # Моно
    sampwidth = 2 # 16-бит
    framerate = 44100 # Частота дискретизации
    nframes = len(audio_data) // (sampwidth * nchannels)
    # Создание файла WAV
    with wave.open('venv/files/output.wav', 'wb') as wav_file:
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.setnframes(nframes)
        wav_file.writeframes(audio_data)

# Функция для распознавания речи
def recognize_speech():    
    model = whisper.load_model('medium')
    transcription = model.transcribe(r"venv/files/output.wav", fp16=False, language='russian') 
    return transcription

def preprocess_text(transcription):
    tokens = word_tokenize(transcription)
    morph = pymorphy3.MorphAnalyzer()    
    tokens = [morph.parse(token)[0].normal_form for token in tokens if token.isalnum() and not token.isdigit()]
    tokens = [token for token in tokens if token not in stopwords.words('russian')]
    return tokens

def command(tokens):
    if 'поиск' in tokens:
        webbrowser.open('https://ya.ru/')
    elif 'видео' in tokens:
        webbrowser.open('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    elif 'блокнот' in tokens:
        subprocess.run(['notepad.exe'])
    else:
        print('Неизвестная команда!')


record_audio()
transcription = recognize_speech()
tokens = preprocess_text(transcription['text'])
print(tokens)
command(tokens)