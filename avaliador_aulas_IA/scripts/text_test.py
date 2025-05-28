import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

arquivo_transcricao = "transcripts/transcricao.txt"

def analisar_transcricao_mock(texto):
    print("🔧 Modo teste ativado: resposta simulada.")
    return """
Resultado da análise:
Clareza: 8/10
Coerência: 9/10
Didática: 7/10
Entendimento: 8/10
Adequação da Linguagem: 9/10

Recomendação: ✅ Aprovado
Justificativa: A aula apresenta boa coerência, linguagem apropriada e é compreensível, apesar de pequenas oportunidades de melhoria na didática.
"""

if __name__ == "__main__":
    try:
        # Lê o conteúdo da transcrição
        with open(arquivo_transcricao, "r", encoding="utf-8") as f:
            texto_transcricao = f.read()

        # Faz análise simulada
        resultado = analisar_transcricao_mock(texto_transcricao)    


        # Salva o resultado
        caminho_resultado = os.path.join("results", "resultado.txt")
        with open(caminho_resultado, "w", encoding="utf-8") as f:
            f.write(resultado)



    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {arquivo_transcricao}")
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
