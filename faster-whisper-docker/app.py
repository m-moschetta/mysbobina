from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette richieste da qualsiasi origine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza il modello Whisper
model = WhisperModel("base", device="cpu")  # Usa "base" come modello

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    # Salva il file caricato temporaneamente
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