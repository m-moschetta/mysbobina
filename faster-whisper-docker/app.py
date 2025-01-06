from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from fastapi.responses import FileResponse
import os
import shutil

app = FastAPI()

# Configura il middleware CORS per permettere richieste da qualsiasi origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permetti richieste da tutte le origini
    allow_credentials=True,
    allow_methods=["*"],  # Consenti tutti i metodi HTTP (GET, POST, ecc.)
    allow_headers=["*"],  # Consenti tutti gli header
)

# Inizializza il modello Whisper
model = WhisperModel("base", device="cpu")  # Puoi cambiare il modello, ad esempio "medium" o "large-v2"

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

        # Salva temporaneamente il file
        temp_file_path = f"/app/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Avvia la trascrizione
        segments, info = model.transcribe(temp_file_path)
        transcription = "\n".join([segment.text for segment in segments])

        # Elimina il file temporaneo
        os.remove(temp_file_path)

        return {
            "language": info.language,
            "transcription": transcription
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la trascrizione: {str(e)}")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve un'icona vuota per evitare errori di richiesta del browser."""
    return FileResponse("favicon.ico") if os.path.exists("favicon.ico") else ""