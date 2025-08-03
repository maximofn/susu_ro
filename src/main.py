from core.agent.susu_ro import SusuRo

def main():
    prompt = "Transcribe el audio data/MicroMachines.mp3"

    susu = SusuRo(whisper_model_size="tiny")

    messages = susu(prompt)
    
    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    main()
