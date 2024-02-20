from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
import uvicorn
import json
from scipy.io.wavfile import write as write_wav
import pathlib
import subprocess
from macros import lang2accent
# Preload models
print("Preloading models...")
preload_models()
# Starting Cache Servers
print("Starting Cache Servers...")
subprocess.Popen(["php","-S","localhost:8080","-t","./cache"],shell=False)
app = FastAPI()
# CORS
print("Setting up CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    textPrompt:str
    character:str
    language:str
print("Setting up routes...")
@app.post("/generate")
def generateAudio(request:RequestBody):
    textPrompt = request.textPrompt
    character = request.character
    language = request.language
    if not pathlib.Path(f"./cache/{language}/{character}").exists():
        pathlib.Path(f"./cache/{language}/{character}").mkdir(parents=True,exist_ok=True)
    if pathlib.Path(f"./cache/{language}/{textPrompt}.wav").exists():
        pass
    else:
        if "AOE" in textPrompt and language == "ja":
            textPrompt = "範囲傷害です"
        else:
            audio_array = generate_audio(textPrompt, prompt=character,language=language,accent=lang2accent[language])
            # Saving a cache
            write_wav(f"./cache/{language}/{character}/{textPrompt}.wav",SAMPLE_RATE,audio_array)
    return json.dumps({"audioURL":f"http://localhost:8080/{language}/{character}/{textPrompt}.wav"})
if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app,host="localhost",port=8000)

