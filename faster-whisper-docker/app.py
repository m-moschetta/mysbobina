from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import os
import subprocess
import shutil
import logging

# Configura logging per il debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Percorsi aggiornati
MODEL_PATH = "/Users/mariomoschetta/Library/Mobile Documents/com~apple~CloudDocs/Mysbobina/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin"
WHISPER_BIN = "/Users/mariomoschetta/Library/Mobile Documents/com~apple~CloudDocs/Mysbobina/whisper.cpp/build/bin/whisper-cli"
TRANSCRIPTION_DIR = "/Users/mariomoschetta/Library/Mobile Documents/com~apple~CloudDocs/Mysbobina/transcriptions"

# Crea la cartella di destinazione per le trascrizioni
os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    """Carica un file audio, lo converte e lo trascrive con Whisper, salvando solo il file di trascrizione."""
    try:
        # Crea directory temporanea
        temp_dir = "/tmp/mysbobina"
        os.makedirs(temp_dir, exist_ok=True)

        # Salva il file originale
        input_path = os.path.join(temp_dir, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Controlla se il file è già in formato WAV 16kHz
        output_path = input_path
        if not file.filename.endswith(".wav"):
            output_path = os.path.join(temp_dir, "converted_audio.wav")
            ffmpeg_command = [
                "ffmpeg", "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                output_path
            ]
            subprocess.run(ffmpeg_command, check=True)

        # Nome file di trascrizione
        transcription_filename = os.path.splitext(file.filename)[0] + ".txt"
        transcription_path = os.path.join(TRANSCRIPTION_DIR, transcription_filename)

        # Esegui Whisper-cli
        command = [
            WHISPER_BIN,
            "-m", MODEL_PATH,
            "-f", output_path,
            "-l", "it",
            "--output-txt"
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Controlla eventuali errori nell'esecuzione
        if result.returncode != 0:
            raise Exception(f"Errore Whisper-cli: {result.stderr}")

        # Verifica che il file di trascrizione sia stato generato
        generated_transcription_file = f"{output_path}.txt"
        if not os.path.exists(generated_transcription_file):
            raise Exception("File di trascrizione non trovato.")

        # Sposta il file di trascrizione nella cartella finale
        shutil.move(generated_transcription_file, transcription_path)

        # Pulizia file temporanei (manteniamo solo la trascrizione)
        os.remove(input_path)
        if input_path != output_path:
            os.remove(output_path)

        return {"transcription_file": transcription_path}

    except subprocess.CalledProcessError as cpe:
        logger.error(f"Errore esecuzione comando esterno: {cpe}")
        raise HTTPException(status_code=500, detail="Errore nell'elaborazione audio.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Errore durante la trascrizione: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Restituisce un'icona vuota se non presente."""
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico")
    return Response(status_code=204)