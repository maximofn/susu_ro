from core.agent.susu_ro import SusuRo
from core.stt.whisper import Whisper

def main():
    # prompt = "Hola, cómo estás?"

    # susu = SusuRo()

    # messages = susu.invoke(prompt)
    
    # for message in messages:
    #     message.pretty_print()

    stt = Whisper()
    audio_path = "data/MicroMachines.mp3"
    result = stt.transcribe(audio_path)
    print(result)
    

if __name__ == "__main__":
    main()
