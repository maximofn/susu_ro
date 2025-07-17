from core.agent.susu_ro import SusuRo
from core.stt.whisper import Whisper

def main():
    prompt = "Transcribe el audio data/MicroMachines.mp3"

    susu = SusuRo()

    messages = susu.invoke(prompt)
    
    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    main()
