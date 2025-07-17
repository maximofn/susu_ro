from core.susu_ro import SusuRo

def main():
    prompt = "Hola, cómo estás?"

    susu = SusuRo()

    messages = susu.invoke(prompt)
    
    for message in messages:
        message.pretty_print()
    

if __name__ == "__main__":
    main()
