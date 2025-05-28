import os
from dotenv import load_dotenv
from openai import OpenAI
import json

def analyze_transcription(text, config):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY não definida.")

    client = OpenAI(api_key=api_key)
    system_message = {
        "role": "system",
        "content": (
            "Você é um avaliador automático de transcrições de aulas. "
            "Analise o texto com base em: coerência, didática, entendimento e adequação do palavreado. "
            "Retorne um JSON com notas de 0 a 10 para cada critério, uma recomendação (Aprovado/Reprovado) "
            "e uma justificativa breve."
        )
    }
    user_message = {"role": "user", "content": f"Analise:\n\n{text}"}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_message, user_message],
        temperature=0.7,
        max_tokens=500
    )
    return json.loads(response.choices[0].message.content)