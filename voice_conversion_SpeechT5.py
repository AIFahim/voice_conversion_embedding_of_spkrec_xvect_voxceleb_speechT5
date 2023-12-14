from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
import numpy as np


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")


from datasets import load_dataset
# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# example = dataset[40]
import torchaudio
x, sr = torchaudio.load("/home/asif/augmentations_experiments/augmentations/test_audio_male.wav")

# sampling_rate = dataset.features["audio"].sampling_rate
inputs = processor(audio=x.squeeze(), sampling_rate=sr, return_tensors="pt")

import torch
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

spek_emb = np.load("/home/asif/augmentations_experiments/speaker_emb/Waz Islamic Sermon.npy")
speaker_embeddings = torch.tensor(spek_emb).unsqueeze(0)

# print(speaker_embeddings)
# assert False

from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)

import soundfile as sf
sf.write("./voice_conversion_SpeechT5/speech_converted_waz_like.wav", speech.numpy(), samplerate=16000)
