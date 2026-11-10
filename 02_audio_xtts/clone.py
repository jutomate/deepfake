import time
import soundfile as sf
import torch
import torchaudio

# Route torchaudio.load through soundfile to bypass torchcodec FFmpeg DLL checks
def soundfile_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = soundfile_load

from TTS.api import TTS

print("[*] Loading XTTSv2 model from local cache...")
t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
print(f"[+] Model loaded in {time.time() - t0:.2f}s")

target_text = (
    "I'm often asked what is your secret to success the shorter answer put inthat work there may be more failures than victory."
)

print("[*] Synthesizing cloned voice...")
t1 = time.time()
tts.tts_to_file(
    text=target_text,
    speaker_wav="reference_voice.wav",
    language="en",
    file_path="cloned_output.wav"
)

print(f"[+] Completed in {time.time() - t1:.2f}s! Saved to cloned_output.wav") 