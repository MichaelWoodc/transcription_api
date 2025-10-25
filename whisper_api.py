# ------------------------------
# FastAPI Whisper Word Splitter with Padding + SSE Status
# ------------------------------

import os
import whisper
import uvicorn
import nest_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from datetime import datetime
import json
import tempfile

nest_asyncio.apply()

# ---------------- Settings ----------------
LANGUAGE = "es"
MODEL_SIZE = "medium"
DEVICE = "cpu"
OUTPUT_DIR = "audio_output"
PADDING_SEC = 0.15  # 150 ms padding before and after word

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
print("✅ Model loaded!")

# ---------------- FastAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve audio files statically
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

# ---------------- Helper Functions ----------------
def save_word_audio(audio_segment, word_index, word_text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_word = "".join(c for c in word_text[:20] if c.isalnum() or c in ("-", "_")).replace(" ", "_") or "word"
    filename = f"{timestamp}_word{word_index}_{safe_word}.wav"
    path = os.path.join(OUTPUT_DIR, filename)
    audio_segment.export(path, format="wav")
    print(f"Saved word audio: {filename}")
    return filename

@app.post("/transcribe_sse")
async def transcribe_audio_sse(file: UploadFile = File(...)):
    # Read the entire upload immediately
    content = await file.read()

    async def event_generator():
        # Notify received
        yield f"data: {json.dumps({'status': 'received', 'filename': file.filename})}\n\n"

        # Create a unique folder for this upload
        timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = "".join(c for c in file.filename if c.isalnum() or c in ("-", "_", "."))
        upload_folder = os.path.join(OUTPUT_DIR, f"{timestamp_folder}_{safe_name}")
        os.makedirs(upload_folder, exist_ok=True)

        # Save original file inside its folder
        save_path = os.path.join(upload_folder, safe_name)
        with open(save_path, "wb") as f:
            f.write(content)
        yield f"data: {json.dumps({'status': 'saved', 'path': save_path})}\n\n"

        # Load audio
        audio = AudioSegment.from_file(save_path)
        yield f"data: {json.dumps({'status': 'transcription_started'})}\n\n"

        # Transcribe
        result = model.transcribe(save_path, language=LANGUAGE, task="transcribe", word_timestamps=True)
        full_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        words_info = []
        word_counter = 0

        for seg in segments:
            for w in seg.get("words", []):
                word_text = w["word"].strip()
                start_ms = max(int((w["start"] - PADDING_SEC) * 1000), 0)
                end_ms = min(int((w["end"] + PADDING_SEC) * 1000), len(audio))
                word_audio = audio[start_ms:end_ms]
                word_filename = save_word_audio(word_audio, word_counter, word_text, upload_folder)
                words_info.append({
                    "index": word_counter,
                    "word": word_text,
                    "start": w["start"],
                    "end": w["end"],
                    "url": f"/audio/{os.path.basename(upload_folder)}/{word_filename}"
                })
                yield f"data: {json.dumps({'status': 'word_processed', 'word_index': word_counter})}\n\n"
                word_counter += 1

        yield f"data: {json.dumps({'status': 'done', 'text': full_text, 'words_count': len(words_info)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Fallback endpoint for non-SSE transcription"""
    try:
        # Read the uploaded file
        content = await file.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Transcribe
        result = model.transcribe(temp_path, language=LANGUAGE, task="transcribe", word_timestamps=True)
        
        # Clean up
        os.unlink(temp_path)
        
        return JSONResponse({
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", [])
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/audio_files")
async def list_audio_files():
    try:
        files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f)) and f.endswith(".wav")]
        files_info = [{"filename": f, "url": f"/audio/{f}"} for f in sorted(files)]
        return JSONResponse({"audio_files": files_info})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/play/{filename}")
async def play_audio(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/wav", filename=filename)
    else:
        return JSONResponse({"error": "File not found"}, status_code=404)

@app.delete("/audio_files")
async def clear_audio_files():
    try:
        for f in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(path):
                os.remove(path)
        return JSONResponse({"message": "All audio files cleared."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {
        "message": "Whisper Spanish Word Transcription API",
        "status": "running",
        "endpoints": {
            "transcribe_sse": "POST /transcribe_sse",
            "transcribe": "POST /transcribe",
            "list_files": "GET /audio_files",
            "play_audio": "GET /play/{filename}",
            "clear_files": "DELETE /audio_files"
        }
    }

# ---------------- Run Server ----------------
if __name__ == "__main__":
    print(f"Server starting on http://127.0.0.1:8000")
    print(f"Word audio files will be saved in: {os.path.abspath(OUTPUT_DIR)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
