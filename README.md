iapd - Video Lesson Analyzer
Este projeto analisa aulas gravadas em vídeo, avaliando critérios como qualidade técnica, linguagem corporal, tom de voz, clareza do roteiro, ritmo, didática e qualidade geral. Ele utiliza ferramentas como MediaPipe, Librosa, Whisper e a API Gemini para gerar relatórios detalhados.
Pré-requisitos

Python 3.8 ou superior
Dependências listadas em requirements.txt
Chave de API da Google Gemini
FFmpeg instalado (necessário para moviepy e whisper)

Instalação

Clone o repositório:
git clone https://github.com/CottaMDM13/IA_Avaliacao_de_aulas
cd iapd


Crie e ative um ambiente virtual (opcional, mas recomendado):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Instale as dependências:
pip install -r requirements.txt

Nota para Windows: Se encontrar problemas com python-magic, instale python-magic-bin:
pip install python-magic-bin==0.4.14


Configure o arquivo .env:Crie um arquivo .env na raiz do projeto com a seguinte estrutura:
GEMINI_API_KEY=sua_chave_de_api_aqui

Obtenha a chave da API Gemini em Google Cloud Console.

Instale o FFmpeg:

Linux: sudo apt-get install ffmpeg
Mac: brew install ffmpeg
Windows: Baixe em FFmpeg.org e adicione ao PATH.



Estrutura do Projeto

src/: Código-fonte principal
analysis/: Módulos para análise de vídeo (video.py), áudio (audio.py), texto (text.py) e métricas (metrics.py)
utils/: Funções auxiliares para carregamento de vídeos (video_loader.py), transcrição (transcriber.py) e geração de relatórios (report.py)


app.py: Aplicação web Flask para upload e visualização de relatórios
main.py: Script CLI para processamento de vídeos
config/settings.yaml: Configurações do projeto (caminhos, limiares, etc.)
templates/: Templates HTML para a interface web
requirements.txt: Dependências do projeto
.env: Arquivo de configuração para a chave da API Gemini

Como Executar
Modo Web (Flask)

Inicie a aplicação:python app.py


Acesse http://localhost:5000 no navegador.
Faça upload de vídeos .mp4 para análise e visualize os relatórios.

Modo CLI

Execute o script main.py com o caminho do vídeo:python main.py caminho/para/video.mp4


Os relatórios serão gerados no diretório especificado em config/settings.yaml.

Configuração
Edite config/settings.yaml para personalizar:

Caminhos de entrada/saída (input_videos, input_audio, output_reports)
Limiares para análise (futuramente, mover limiares de audio.py e video.py para cá)

Exemplo:
paths:
  input_videos: videos/input
  input_audio: audio/input
  output_reports: reports/output
  output_transcripts: transcripts/output
thresholds:
  face_visibility: 0.8

Dependências Principais

flask: Framework web
mediapipe: Análise de gestos e expressões faciais
librosa: Análise de áudio
whisper: Transcrição de áudio
google-generativeai: Geração de comentários com Gemini
python-magic: Validação de tipo de arquivo (MIME)

Notas

Tamanho máximo de upload: 100MB (configurado em app.py).
Formatos suportados: Apenas .mp4.
Relatórios: Gerados em JSON e DOCX, armazenados em reports/output e no banco SQLite (reports.db).

Contribuição

Fork o repositório.
Crie uma branch: git checkout -b minha-melhoria.
Faça commit das alterações: git commit -m "Descrição da melhoria".
Envie para o repositório remoto: git push origin minha-melhoria.
Abra um Pull Request.

Licença
MIT License
