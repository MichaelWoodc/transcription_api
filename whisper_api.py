### Best so far
# ------------------------------
# FastAPI Whisper Word Splitter with Padding
# ------------------------------

import os
import tempfile
import whisper
import uvicorn
import nest_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from datetime import datetime

nest_asyncio.apply()

# ---------------- Settings ----------------
LANGUAGE = "es"
MODEL_SIZE = "medium"
DEVICE = "cpu"
OUTPUT_DIR = "audio_output"
PADDING_SEC = 0.15  # 100 ms padding before and after word

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
print("Model loaded!")

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

# ---------------- Endpoints ----------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    tmp_path = None
    try:
        print(f"Received file: {file.filename} ({file.content_type})")
        content = await file.read()
        suffix = "." + file.filename.split(".")[-1] if "." in file.filename else ".wav"

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Convert to WAV mono 16kHz
        print("Converting audio to WAV mono 16kHz...")
        audio = AudioSegment.from_file(tmp_path).set_channels(1).set_frame_rate(16000)
        wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        print(f"Converted audio saved temporarily as {wav_path}")

        # Transcribe with word-level timestamps
        print("Transcribing with Whisper...")
        result = model.transcribe(wav_path, language=LANGUAGE, task="transcribe", word_timestamps=True)
        full_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        words_info = []
        word_counter = 0

        # Split audio by words with padding
        for seg in segments:
            for w in seg.get("words", []):
                word_text = w["word"].strip()
                start_ms = max(int((w["start"] - PADDING_SEC) * 1000), 0)
                end_ms = min(int((w["end"] + PADDING_SEC) * 1000), len(audio))
                word_audio = audio[start_ms:end_ms]
                word_filename = save_word_audio(word_audio, word_counter, word_text)
                words_info.append({
                    "index": word_counter,
                    "word": word_text,
                    "start": w["start"],
                    "end": w["end"],
                    "url": f"/audio/{word_filename}"
                })
                word_counter += 1

        # Cleanup temp
        try:
            os.remove(tmp_path)
            os.remove(wav_path)
        except:
            pass

        print(f"Transcription complete: {full_text}")
        print(f"{len(words_info)} word files created.")

        return JSONResponse({"text": full_text, "words": words_info})

    except Exception as e:
        print(f"Error processing file: {e}")
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
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
