# Usa un'immagine base Python con una versione stabile
FROM python:3.11-slim

# Installa pip e le librerie necessarie
RUN pip install --upgrade pip
RUN pip install faster-whisper fastapi uvicorn python-multipart

# Installa le dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y \
    ffmpeg \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file dell'applicazione nel container
COPY app.py /app/app.py

# Aggiorna pip e installa i pacchetti necessari
RUN pip install --upgrade pip
RUN pip install faster-whisper fastapi uvicorn

# Espone la porta necessaria per FastAPI
EXPOSE 8000

# Comando predefinito per avviare il server FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]