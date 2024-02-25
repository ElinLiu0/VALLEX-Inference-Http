from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.generation import SAMPLE_RATE, generate_audio, preload_models,generate_audio_from_long_text
import uvicorn
import json
from scipy.io.wavfile import write as write_wav
import pathlib
import subprocess
from macros import lang2accent
import torch
import os

os.environ['CURL_CA_BUNDLE'] = ''
# 如果允许的话，可以解除以下代码的注释
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 如果有多个GPUs时，可以使用逗号分隔GPU ID


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
    noaccent:bool
    longprompt:bool
print("Setting up routes...")
@app.post("/generate")
async def generateAudio(request:RequestBody):
    textPrompt = request.textPrompt
    character = request.character
    language = request.language
    noaccent = request.noaccent
    longpromptMode = request.longprompt
    executionMode = "long" if longpromptMode else "short"
    if language not in lang2accent.keys():
        return {"error":"Language not support","code":"404"}
    if not pathlib.Path(f"./presets/{character}.npz").exists():
        return {"error":"Character not found","code":"404"}
    if not pathlib.Path(f"./cache/{language}/{character}/{executionMode}").exists():
        pathlib.Path(f"./cache/{language}/{character}/{executionMode}").mkdir(parents=True,exist_ok=True)
    if pathlib.Path(f"./cache/{language}/{character}/{textPrompt}.wav").exists():
        return {"audioURL":f"http://localhost:8080/{language}/{character}/{executionMode}/{textPrompt}.wav","code":"200"}
    else:
        if "AOE" in textPrompt and language == "ja":
            textPrompt = "範囲傷害です"
        else:
            try:
                if longpromptMode:
                    audio_array = generate_audio_from_long_text(textPrompt, prompt=character,language=language,accent=lang2accent[language] if noaccent == False else "no-accent")
                else:
                    audio_array = generate_audio(textPrompt, prompt=character,language=language,accent=lang2accent[language] if noaccent == False else "no-accent")
                # Saving a cache
                write_wav(f"./cache/{language}/{character}/{executionMode}/{textPrompt}.wav",SAMPLE_RATE,audio_array)
                torch.cuda.synchronize() # 同步所有的GPU Stream
                torch.cuda.empty_cache() # 清空所有的GPU缓存
            except Exception as e:
                return {"error":str(e),"code":"500"}
    return {"audioURL":f"http://localhost:8080/{language}/{character}/{executionMode}/{textPrompt}.wav","code":"200"}
if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app,host="localhost",port=8000)

