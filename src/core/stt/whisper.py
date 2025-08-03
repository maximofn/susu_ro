import whisper
import torch
import warnings

class Whisper:
    def __init__(self, model_size: str = "large", device: str = "auto"):
        """
        Initialize the Whisper model.

        Args:
            model_size (str, optional): The size of the Whisper model. Defaults to "large".
            device (str, optional): The device to use for the model. Options: "auto", "cpu", "cuda", "mps". Defaults to "auto".
        """
        # Auto-detect the best available device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"  # Use Metal Performance Shaders on Mac M1/M2
            elif torch.cuda.is_available():
                device = "cuda"  # Use CUDA if available
            else:
                device = "cpu"   # Fallback to CPU
        
        # Try to load the model with the specified device, with fallback for MPS and CUDA compatibility issues
        self.device = device
        try:
            self.model = whisper.load_model(model_size, device=device)
        except (NotImplementedError, RuntimeError) as e:
            if device == "mps":
                warnings.warn(
                    f"⚠️ MPS backend failed due to compatibility issues. Falling back to CPU.\n"
                    f"Error: {str(e)[:100]}..."
                )
                self.device = "cpu"
                self.model = whisper.load_model(model_size, device="cpu")
                print(f"✅ Whisper model loaded successfully on CPU (MPS fallback)")
            elif device == "cuda":
                warnings.warn(
                    f"⚠️ CUDA backend failed due to compatibility issues. Falling back to CPU.\n"
                    f"Error: {str(e)[:100]}..."
                )
                self.device = "cpu"
                self.model = whisper.load_model(model_size, device="cpu")
                print(f"✅ Whisper model loaded successfully on CPU (CUDA fallback)")
            else:
                raise e

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
            "fp16": True if self.device in ["cuda", "mps"] else False,
        }
        result = self.model.transcribe(audio_path, **decode_options)
        return result["text"]