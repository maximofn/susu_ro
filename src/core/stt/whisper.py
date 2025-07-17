import whisper

class Whisper:
    def __init__(self, model_size: str = "large", device: str = "cpu"):
        """
        Initialize the Whisper model.

        Args:
            model_size (str, optional): The size of the Whisper model. Defaults to "large".
            device (str, optional): The device to use for the model. Defaults to "cpu".
        """
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path: str, language: str = None) -> str:
        """
        Transcribe an audio file using the Whisper model.

        Args:
            audio_path (str): The path to the audio file to transcribe.
            language (str, optional): The language of the audio file. Defaults to None.

        Returns:
            str: The transcribed text.
        """
        decode_options = {
            "language": language,
            "fp16": True if self.model.device == "cuda" else False,
        }
        result = self.model.transcribe(audio_path, **decode_options)
        return result["text"]