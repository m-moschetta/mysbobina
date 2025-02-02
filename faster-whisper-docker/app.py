from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import subprocess
import shutil

app = FastAPI(redirect_slashes=False)

# Configura il middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["ngrok-skip-browser-warning"]
)

# Percorsi per whisper.cpp
WHISPER_DIR = "/Users/mariomoschetta/Library/Mobile Documents/com~apple~CloudDocs/Mysbobina/whisper.cpp"
MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo-q8_0.bin")
WHISPER_BIN = os.path.join(WHISPER_DIR, "main")


@app.get("/")
def read_root():
    """Messaggio di benvenuto per la radice."""
    return {"message": "Benvenuto su MySbobina! Usa l'endpoint /upload per trascrivere i file audio."}

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    """Carica un file audio e restituisci la trascrizione."""
    try:
        # Verifica che il file sia un audio
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Il file caricato non è un audio valido.")

        # Crea una directory temporanea se non esiste
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Salva temporaneamente il file
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File salvato temporaneamente in: {temp_file_path}")

        # Comando per eseguire whisper.cpp con Core ML (ANE)
        command = [
            WHISPER_BIN,
            "-m", MODEL_PATH,
            "-f", temp_file_path,
            "--coreml",
            "-otxt"
        ]

        # Esegui la trascrizione
        result = subprocess.run(command, 
                              capture_output=True, 
                              text=True,
                              cwd=WHISPER_DIR)

        if result.returncode != 0:
            raise Exception(f"Errore whisper.cpp: {result.stderr}")

        # Leggi il risultato della trascrizione
        output_file = f"{temp_file_path}.txt"
        with open(output_file, "r") as f:
            transcription = f.read()

        # Pulizia file temporanei
        os.remove(temp_file_path)
        os.remove(output_file)

        return {
            "transcription": transcription
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Errore durante la trascrizione: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore durante la trascrizione: {str(e)}")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve un'icona vuota per evitare errori di richiesta del browser."""
    return FileResponse("favicon.ico") if os.path.exists("favicon.ico") else ""
