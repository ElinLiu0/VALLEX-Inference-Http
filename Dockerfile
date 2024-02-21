FROM python:3.10
RUN apt-get update -y
RUN apt-get install git git-lfs wget php-cli -y
RUN git lfs install
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
RUN git clone https://github.com/Plachtaa/VALL-E-X
RUN mkdir -p ./VALL-E-X/checkpoints
RUN mkdir -p ./VALL-E-X/whisper
RUN mkdir -p ./VALL-E-X/logs
RUN pip install -r ./VALL-E-X/requirements.txt
RUN wget https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt -P ./VALL-E-X/checkpoints
RUN wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -P ./VALL-E-X/whisper
RUN wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Graha-Finetuned.npz -P ./VALL-E-X/presets
RUN wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Hythloadeus-Finetuned.npz -P ./VALL-E-X/presets
RUN wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Ryne-Finetuned.npz -P ./VALL-E-X/presets
RUN wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Venat-Finetuned.npz -P ./VALL-E-X/presets
RUN pip install fastapi uvicorn pydantic
RUN wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/HTTPServer.py
EXPOSE 8000
EXPOSE 8080
CMD [nohup uvicorn HTTPServer:app --reload > ./logs/server.log &]
