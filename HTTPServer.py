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
import torch
import os


# 如果允许的话，可以解除以下代码的注释
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 如果有多个GPUs时，可以使用逗号分隔GPU ID


# Preload models
print("Preloading models...")
preload_models()
# Starting Cache Servers
print("Starting Cache Servers...")
subprocess.Popen(["php","-S","0.0.0.0:8080","-t","./cache"],shell=False)
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
    noaccent:bool
print("Setting up routes...")
@app.post("/generate")
async def generateAudio(request:RequestBody):
    textPrompt = request.textPrompt
    character = request.character
    language = request.language
    noaccent = request.noaccent
    if not pathlib.Path(f"./presets/{character}.npz").exists():
        return json.dumps({"error":"Character not found","code":"404"})
    if not pathlib.Path(f"./cache/{language}/{character}").exists():
        pathlib.Path(f"./cache/{language}/{character}").mkdir(parents=True,exist_ok=True)
    if pathlib.Path(f"./cache/{language}/{textPrompt}.wav").exists():
        return json.dumps({"audioURL":f"http://localhost:8080/{language}/{character}/{textPrompt}.wav","code":"200"})
    else:
        if "AOE" in textPrompt and language == "ja":
            textPrompt = "範囲傷害です"
        else:
            try:
                audio_array = generate_audio(textPrompt, prompt=character,language=language,accent=lang2accent[language] if noaccent == False else "no-accent")
                # Saving a cache
                write_wav(f"./cache/{language}/{character}/{textPrompt}.wav",SAMPLE_RATE,audio_array)
                torch.cuda.synchronize() # 同步所有的GPU Stream
                torch.cuda.empty_cache() # 清空所有的GPU缓存
            except Exception as e:
                return json.dumps({"error":str(e),"code":"500"})
    return json.dumps({"audioURL":f"http://0.0.0.0:8080/{language}/{character}/{textPrompt}.wav","code":"200"})
if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app,host="localhost",port=8000)

