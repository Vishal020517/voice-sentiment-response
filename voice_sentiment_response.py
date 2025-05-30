import os
import torch
import re
import sounddevice as sd
import numpy as np
import torchaudio
import google.generativeai as genai
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import pipeline

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAw2NboM9zg9YYgJH_icLo2RWSpYIOP19s")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

nltk.download('stopwords')
nltk.download('wordnet')

DURATION = 5
SAMPLE_RATE = 16000

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

print("Loading dataset and training classifier...")
dataset = load_dataset("dair-ai/emotion", split="train")
X_train = [preprocess_text(item['text']) for item in dataset]
y_train = [item['label'] for item in dataset]
label_names = dataset.features['label'].names

emotion_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, solver='liblinear'))
])
emotion_model.fit(X_train, y_train)
print("Emotion classifier ready.\n")

def generate_emotional_response(text, emotion):
    instructions = {
        "joy": "You are positive and enthusiastic. Respond with excitement.",
        "sadness": "You are empathetic and supportive. Offer comfort.",
        "anger": "You are calm and neutral. De-escalate the emotion.",
        "surprise": "You are curious. Express amazement and interest.",
        "fear": "You are reassuring. Reduce fear and give confidence.",
        "disgust": "You are understanding. React neutrally.",
    }
    system_instruction = instructions.get(emotion, "You are polite and helpful.")
    user_prompt = f"The user expressed '{emotion}': '{text}'. Reply in english and appropriately."
    
    try:
        chat = gemini_model.start_chat()
        response = chat.send_message(f"SYSTEM INSTRUCTION: {system_instruction}\nUSER INPUT: {user_prompt}")
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

print("Recording... Speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
print("Recording finished.")

waveform = torch.from_numpy(audio.T)
torchaudio.save("live_input.wav", waveform, SAMPLE_RATE)

print("Transcribing...")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
transcribed = asr("live_input.wav")["text"]
print(f"Transcribed Text: {transcribed}")

processed_input = preprocess_text(transcribed)
predicted_label = emotion_model.predict([processed_input])[0]
emotion = label_names[predicted_label]
print(f"Detected Emotion: {emotion}")

response = generate_emotional_response(transcribed, emotion)
print(f"\nAI Response:\n{response}")
