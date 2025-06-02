import os
import torch
import sounddevice as sd
import soundfile as sf
import whisper
from transformers import pipeline
from openwakeword.model import Model
from tts import synthesize_speech
import time

# Config
AUDIO_FILE = "input.wav"
RECORD_SECONDS = 5
SAMPLING_RATE = 16000
WAKEWORDS = ["hey computer"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Init
print("Loading models...")

wakeword_model = Model()
wakeword_model.load_models()

whisper_model = whisper.load_model("base")

llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct", device=0 if torch.cuda.is_available() else -1)

print("✅ Assistant Ready")

def listen_and_record(filename, duration=RECORD_SECONDS, samplerate=SAMPLING_RATE):
    print("🎙 Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    return filename

def detect_wakeword():
    print("👂 Waiting for wake word...")
    duration = 1.0
    while True:
        audio = sd.rec(int(duration * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
        sd.wait()
        audio = audio.squeeze()
        detected = wakeword_model.predict(audio, SAMPLING_RATE)
        for word, prob in detected.items():
            if word.lower() in WAKEWORDS and prob > 0.7:
                print(f"✅ Wake word '{word}' detected!")
                return

def transcribe_audio(filename):
    result = whisper_model.transcribe(filename)
    return result["text"]

def query_llm(prompt):
    response = llm(f"<s>[INST] {prompt} [/INST]", max_new_tokens=150, do_sample=True)
    return response[0]["generated_text"].split("[/INST]")[-1].strip()

# Main loop
while True:
    detect_wakeword()
    listen_and_record(AUDIO_FILE)
    query = transcribe_audio(AUDIO_FILE)
    print("🗣 You said:", query)

    if query.lower() in ["exit", "quit", "stop"]:
        print("👋 Goodbye!")
        synthesize_speech("Goodbye!", blocking=True)
        break

    response = query_llm(query)
    print("🤖 Assistant:", response)
    synthesize_speech(response, blocking=True)
