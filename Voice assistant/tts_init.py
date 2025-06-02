import torch
from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN

class SpeechBrainTTS:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.fastspeech2 = FastSpeech2.from_hparams(
            source="speechbrain/tts-fastspeech2-ljspeech",
            savedir="pretrained_models/tts-fastspeech2-ljspeech"
        ).to(self.device)

        self.hifigan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="pretrained_models/tts-hifigan-ljspeech"
        ).to(self.device)

        if self.use_gpu:
            self.fastspeech2.model.half()
            self.hifigan.model.half()

        self.sr = 22050

    def tts_inference(self, text: str) -> torch.Tensor:
        input_list = [text]
        mel_output, _, _, _ = self.fastspeech2.encode_text(input_list)
        mel_output = mel_output.half() if self.use_gpu else mel_output
        mel_output = mel_output.to(self.device)
        waveforms = self.hifigan.decode_batch(mel_output)
        return waveforms[0]

use_gpu = True
TTS_ENGINE = SpeechBrainTTS(use_gpu=use_gpu)
