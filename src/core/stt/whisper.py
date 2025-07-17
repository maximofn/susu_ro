import whisper

class Whisper:
    def __init__(self, model_size: str = "large", device: str = "cpu"):
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path: str, language: str = None) -> str:
        decode_options = {
            "language": language,
            "fp16": True if self.model.device == "cuda" else False,
        }
        result = self.model.transcribe(audio_path, **decode_options)
        return result["text"]