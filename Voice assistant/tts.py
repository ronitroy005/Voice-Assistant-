import re
import torch
import numpy as np
import sounddevice as sd
from num2words import num2words
from tts_init import TTS_ENGINE

IS_SPEAKING = False

def stop_playback():
    global IS_SPEAKING
    sd.stop()
    IS_SPEAKING = False

def expand_numbers_in_text(text: str) -> str:
    return re.sub(r'\b\d+\b', lambda m: num2words(int(m.group())), text)

def clean_and_check_text(text: str) -> str:
    expanded = expand_numbers_in_text(text)
    expanded = re.sub(r'[^a-zA-Z0-9\s,.?!-]', '', expanded).strip()
    return expanded if re.search(r'[a-zA-Z0-9]', expanded) else ""

def synthesize_speech(text: str, blocking: bool = False):
    global IS_SPEAKING
    clean_text = clean_and_check_text(text)
    if not clean_text:
        print("TTS: skipping, no valid text.")
        return
    try:
        wave_tensor = TTS_ENGINE.tts_inference(clean_text)
        wave_tensor = wave_tensor.float() if wave_tensor.dtype == torch.half else wave_tensor
        audio_np = wave_tensor.cpu().detach().squeeze().numpy().astype(np.float32)
        IS_SPEAKING = True
        sd.play(audio_np, TTS_ENGINE.sr)
        if blocking:
            sd.wait()
            IS_SPEAKING = False
    except Exception as e:
        print(f"TTS Error: {e}")
