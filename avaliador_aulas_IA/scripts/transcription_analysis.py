import os
from dotenv import load_dotenv
from openai import OpenAI

def analisar_transcricao(texto):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

    client = OpenAI(api_key=api_key)

    system_message = {
        "role": "system",
        "content": (
            "Você é um avaliador automático de transcrições de aulas. "
            "Analise o texto com base nos critérios: coerência, didática, entendimento e adequação do palavreado. "
            "Retorne um JSON com notas de 0 a 10 para cada critério e uma recomendação final (Aprovado ou Reprovado), "
            "junto com uma justificativa breve."
        )
    }

    user_message = {
        "role": "user",
        "content": f"Analise a seguinte transcrição:\n\n{texto}"
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_message, user_message],
        temperature=0.7,
        max_tokens=500,
        n=1
    )

    resultado = response.choices[0].message.content
    return resultado

if __name__ == "__main__":
    load_dotenv()  # Carrega as variáveis do arquivo .env   

    arquivo_transcricao = "transcripts/transcricao.txt"
    with open(arquivo_transcricao, "r", encoding="utf-8") as f:
        texto_transcricao = f.read()

    resultado = analisar_transcricao(texto_transcricao)
    print("Resultado da análise da transcrição:")
    print(resultado)
