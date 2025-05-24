from fastapi import FastAPI
from record import record_audio
from model_asr_nlp import asr_pipeline, nlp_pipeline
import uvicorn
import time

app = FastAPI()

@app.get("/asr")
def transcribe():
    audio_path, start_time = record_audio()

    if not audio_path:
        return {"error": "Âm thanh không đạt yêu cầu. Hãy thử lại."}

    text = asr_pipeline(audio_path)
    result = nlp_pipeline(text["text"])
    print(f"Result: {result}")
    duration = round(time.time() - start_time, 2)
    return {
        "result": result,
        "inference_time_sec": duration
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)