#! /usr/bin/env python
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script initiating_mlflow_tracking.py
====================================
Para usar este script, primeiro execute o comando abaixo para instanciar o
MLflow Server. E em outro terminal execute o script para interagir na
linha de comando.

RUN
---
uvx mlflow server --host 0.0.0.0 --cors-allowed-origins "*"

Depois, acesse usando o IP do WSL2: http://172.31.94.105:5000

OBS:

     Onde você está	        |      URL para acessar
Dentro do WSL2 (terminal)	| http://localhost:5000 ou http://127.0.0.1:5000
No Windows (navegador)	    | http://localhost:5000 (via port-forwarding) ou http://172.31.94.105:5000 (IP do WSL2)
"""
import mlflow
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv()) # read local .env file
#openai.api_key  = os.environ['OPENAI_API_KEY']
#Eddy_key_openai  = os.environ['OPENAI_API_KEY']
# Specify the tracking URI for the MLflow server.
mlflow.set_tracking_uri("http://172.31.94.105:5000/") # http://172.31.94.105:5000/   ou   http://localhost:5000

# Specify the experiment you just created for your GenAI application.
mlflow.set_experiment("My_Application_Eddy1")

# Enable automatic tracing for all OpenAI API calls.
mlflow.openai.autolog()

client = OpenAI()


def chat_completion(
    user_message: str,
    system_message: str = "Você é um assistente educado que responde às perguntas do usuário.",
    model: str = "o4-mini",
) -> str:
    """
    Realiza uma chamada ao OpenAI Chat Completion com rastreamento MLflow.

    Args:
        user_message: A mensagem do usuário
        system_message: A mensagem de sistema (contexto do assistente)
        model: O modelo OpenAI a ser usado

    Returns:
        A resposta do modelo
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Assistente Interativo - Digite '/sair' para encerrar\n")

    while True:
        try:
            user_input = input("Você: ").strip()

            if user_input.lower() in ["/sair", "/quit"]:
                print("Até logo!")
                break

            if user_input:
                resposta = chat_completion(user_message=user_input)
                print(f"Assistente: {resposta}\n")

        except KeyboardInterrupt:
            print("\nAté logo!")
            break
