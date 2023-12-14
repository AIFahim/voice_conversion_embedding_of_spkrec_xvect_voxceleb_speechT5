from TTS.api import TTS
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False)
tts.voice_conversion_to_file(source_wav="/home/asif/augmentations_experiments/augmentations/test_audio_male.wav", target_wav="/home/asif/augmentations_experiments/Waz Islamic Sermon.wav", file_path="/home/asif/augmentations_experiments/voice_conversion_SpeechT5/output.wav")