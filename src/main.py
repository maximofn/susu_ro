from core.agent.susu_ro import SusuRo

def main():
    prompt = "Transcribe el audio /Users/macm1/Documents/proyectos/susu_ro/data/slprl__sTinyStories_01.wav que está en inglés"

    whisper_model_size = "tiny"
    whisper_device = "cpu"   # Use CPU for Whisper to avoid MPS compatibility issues
    chat_model = "qwen3:4b"
    # chat_model = "openai/gpt-4.1"
    chat_reasoning = True
    enable_streaming = True

    susu = SusuRo(
        chat_model=chat_model,
        chat_reasoning=chat_reasoning,
        whisper_model_size=whisper_model_size,
        whisper_device=whisper_device,
        enable_streaming=enable_streaming,
    )

    messages = susu(prompt)
    
    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    main()
