### 环境准备

● Python3.11.0或稍低版本（3.9~3.11内）的Python解释器（验证环境为3.11.0，基于Conda）

```bash
# 下载conda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
# 为Conda.sh添加临时权限
sudo chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
# 执行bash
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

按住回车直至你看到以下信息时输入yes，注意不要按的太死以防止直接跳过退出安装脚本：

```bash
Do you accept the license terms? [yes|no]
[no] >>>
```

当输入yes后，你会看到如下信息：

```bash
Anaconda3 will now be installed into this location:
/home/elin/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/elin/anaconda3] >>>
```

如果你的系统用户是非root用户，则conda默认会安装在：/home/<`$USERNAME`>/anaconda3/下，反之是root/anaconda3下。其中`$USERNAME`是你当前的Linux用户名称，在本例中为：elin。

```bash
source ~/.bashrc
```

如果在使用source命令后，conda命令仍然失效，请打开~/.bashrc手动编辑：

```bash
# 打开~/.bashrc
vim ~/.bashrc
# 按下caps lock+G跳转至末尾，并添加如下段
export PATH=/home/<$USERNAME>/anaconda3/bin:$PATH
或
export PATH=/root/anaconda3/bin:$PATH
# 按下:wq保存并退出
# 再次执行：
source ~/.bashrc
# 输入如下：
conda --version
# 当环境变量加载成功后，您应该得到如下输出：
conda 23.7.4
```

至此，conda安装完成。接下来我们准备一个新的conda env（conda环境）:

```bash
# 使用conda create -n命令创建一个新的conda环境，这么做是为了防止base环境被彻底污染
conda create -n <$NEW_ENV_NAME> -y # 这里的$NEW_ENV_NAME可以设置为任意值，以本例为例可以设置为valle
```

当命令执行完毕后，可使用：

```bash
conda env list
```

来验证conda环境是否创建成功，当我们看到包含如下字样的结果时：

```bash
valle                 *  /home/elin/anaconda3/envs/valle
```

则证明该环境已经创建完成，接下来我们激活该环境：

```bash
conda activate valle
```

当我们的终端变成了：

```bash
(valle) elin@ElinWorkstation:~$
```

即说明我们成功激活了valle环境，但该环境下并无python的二进制可执行程序，因此我们需要安装python3.11.0：

```bash
conda install python==3.11.0 -y
```

> 当上述处理方式速度过慢时，可尝试使用mamba处理器进行环境重构建，但前提是一定要没有重名的conda env，如果要覆盖原有的env可以尝试对其先进行删除：
> 
> ```bash
> conda env remove --name <$OLD_ENV_NAME> -y
> ```

> 接下来，我们可以配置conda镜像源为国内镜像源，具体方式可以参考[Anaconda 源使用帮助 — USTC Mirror Help 文档](https://mirrors.ustc.edu.cn/help/anaconda.html)，由于mamba会自动寻找延迟最低的有效仓库（实现原理类似于Redhat Linux上的yum）
> 
> 在完成源配置后，我们可以使用下面的命令来调用mamba处理器创建新的环境：
> 
> ```bash
> conda create --solver=libmamba -n <$NEW_ENV_NAME> python=3.11 -y
> ```

当命令执行完毕后，我们输入：

```bash
python --version
```

如果得到了如下结果：

```bash
(valle) elin@ElinWorkstation:~$ python --version
Python 3.11.0
(valle) elin@ElinWorkstation:~$
```

至此，Python安装完成。

---

● Ubuntu20.04及更高版本或Debian Bullseye及Bookworm（不建议使用Bookworm，存在依赖错误风险，测试环境为Ubuntu22.04）

● 8000以及8080端口可用（可在后期代码中进行更改）

● php-cli

```bash
# 使用如下命令安装php-cli
sudo apt-get install php-cli -y
# 验证php-cli是否安装成功
php --version
# 当得到以下结果时证明php-cli安装成功
(valle) elin@ElinWorkstation:~$ php --version
PHP 8.1.2-1ubuntu2.14 (cli) (built: Aug 18 2023 11:41:11) (NTS)
Copyright (c) The PHP Group
Zend Engine v4.1.2, Copyright (c) Zend Technologies
    with Zend OPcache v8.1.2-1ubuntu2.14, Copyright (c), by Zend Technologies
(valle) elin@ElinWorkstation:~$
```

● git以及git-lfs

```bash
# 使用如下命令安装git及git-lfs
sudo apt-get install git git-lfs -y
# 验证git是否安装成功
git --version
# 当得到如下结果时表示git安装成功：
(valle) elin@ElinWorkstation:~$ git --version
git version 2.34.1
(valle) elin@ElinWorkstation:~$
## 请注意！！！此时git-lfs并没有实际安装成功，请使用
git lfs install
# 当看到以下字样时：
Git LFS initialized.
# 则git-lfs安装完成
```

● 当系统中安装了CUDA GPU时应安装如下SDKs：

    ○ CUDA ToolKit 11.8（测试使用）

```bash
# 下载cuda-ubuntu2204.pin apt秘钥文件
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# 将秘钥移动至apt根目录下
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# 下载CUDA .deb安装包
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
# 应用.deb安装包
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
# 复制秘钥环
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
# 加载本地镜像源
sudo apt-get update
# 安装CUDA
sudo apt-get -y install cuda
# 请注意！此时cuda并不会添加到环境变量，请自行修改~/.bashrc
vim ~/.bashrc
# 使用capslock+G跳转至末尾行并添加
export PATH=/usr/local/cuda/bin:$PATH
# 按下esc,按下:wq保存
# 令.bashrc生效
source ~/.bashrc
# 使用nvcc --version验证cuda是否添加到环境变量中
# 当得到以下结果时代表CUDA安装完成
(valle) elin@ElinWorkstation:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
(valle) elin@ElinWorkstation:~$
```

    ○ cuDNN 8.7.0.84

```bash
# 下载cudnn二进制压缩包
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
# 解压二进制压缩包
tar -xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
# 将当前工作目录切换到lib下
cd cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib
# 将所有的文件拷贝到/usr/local/cuda/targets/x86_64-linux/lib/
sudo cp -a *.* /usr/local/cuda/targets/x86_64-linux/lib/
# 使用ldconfig加载
ldconfig
# 使用ldconfig -p | grep检查cudnn是否加载
ldconfig -p | grep cudnn
# 当得到如下结果时表示cudnn安装成功
(valle) elin@ElinWorkstation:~/cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib$ ldconfig -p | grep cudnn
        libcudnn_ops_train.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
        libcudnn_ops_train.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_ops_train.so
        libcudnn_ops_infer.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
        libcudnn_ops_infer.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_ops_infer.so
        libcudnn_cnn_train.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
        libcudnn_cnn_train.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_cnn_train.so
        libcudnn_cnn_infer.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
        libcudnn_cnn_infer.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_cnn_infer.so
        libcudnn_adv_train.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
        libcudnn_adv_train.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_adv_train.so
        libcudnn_adv_infer.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
        libcudnn_adv_infer.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn_adv_infer.so
        libcudnn.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so.8         libcudnn.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so
```

    ○ TensorRT 8.6.1.6

```bash
# 使用wget下载TensorRT二进制压缩包
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# 解压TensorRT二进制包
tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
# 将工作目录切换到TensorRT/lib下
cd TensorRT-8.6.1.6/lib
# 将所有的文件拷贝到/usr/local/cuda/targets/x86_64-linux/lib/
sudo cp -a *.* /usr/local/cuda/targets/x86_64-linux/lib/
# 使用ldconfig加载
ldconfig
# 使用ldconfig -p | grep检查TensorRT是否加载
ldconfig -p | grep libnvinfer
# 当得到如下结果时表示TensorRT安装成功
(valle) elin@ElinWorkstation:~/TensorRT-8.6.1.6/lib$ ldconfig -p | grep libnvinfer
        libnvinfer_vc_plugin.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_vc_plugin.so.8
        libnvinfer_vc_plugin.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_vc_plugin.so
        libnvinfer_plugin.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_plugin.so.8
        libnvinfer_plugin.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_plugin.so
        libnvinfer_lean.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_lean.so.8
        libnvinfer_lean.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_lean.so
        libnvinfer_dispatch.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_dispatch.so.8
        libnvinfer_dispatch.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer_dispatch.so
        libnvinfer.so.8 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer.so.8
        libnvinfer.so (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libnvinfer.so
(valle) elin@ElinWorkstation:~/TensorRT-8.6.1.6/lib$
```

● [VALL-E-X代码库]([GitHub - Plachtaa/VALL-E-X: An open source implementation of Microsoft&#39;s VALL-E X zero-shot TTS model. Demo is available in https://plachtaa.github.io](https://github.com/Plachtaa/VALL-E-X))

### 依赖

详情见：`VALL-E-X/requirements.txt`

同时，为了服务器能够顺利运行，请在安装好既定依赖后安装uvicorn和fastapi

```python
# 确保pip使用的是中科大镜像
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
# 安装依赖
pip install -r requirements.txt uvicorn fastapi
```

### 模型准备

根据VALL-E-X原作者的提示，请在VALL-E-X项目目录下准备如下目录：

```bash
mkdir -p ./VALL-E-X/whisper ./VALL-E-X/checkpoints
```

默认情况下whisper目录是用不上，只有当你使用：

```bash
python launch-ui.py
```

启动了Gradio WebUI页面时，该目录才会生效，其将会读取该目录下的medium.pt模型来实现短程语料模型的微调。

分别使用：

```bash
wget https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt -P ./VALL-E-X/checkpoints
wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -P ./VALL-E-X/whisper
```

来保存VALL-E-X模型和whisper模型。

下载预制Numpy矩阵到preset目录下：

```bash
wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Graha-Finetuned.npz -P ./VALL-E-X/presets
wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Hythloadeus-Finetuned.npz -P ./VALL-E-X/presets
wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Ryne-Finetuned.npz -P ./VALL-E-X/presets
wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Venat-Finetuned.npz -P ./VALL-E-X/presets
```

### 在macro.py里添加lang2accent字典

```python
lang2accent = {
    'en': 'English',
    'zh': '中文',
    'ja': '日本語',
    'mix': 'Mix'
}

```

如果使用CPU运行，则需要将所有的`torch.device("cuda", 0)`替换为`torch.device("cpu")`

还得把`HTTPServer.py`里的`torch.cuda.synchronize()`和`torch.cuda.empty_cache()`注释掉。

### 服务启动

下载`HTTPServer.py`并保存到VALL-E-X项目目录下：

```bash
wget https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/HTTPServer.py
```

创建logs和cache目录到VALL-E-X目录下：

```bash
mkdir -p ./logs ./cache
mkdir -p ./cache/en
mkdir -p ./cache/ja
mkdir -p ./cache/zh
```

同时，cache目录应保持如下结构：

```bash
.
├── en
│   ├── AOE.wav
│   └── paimon
│       └── Hi travller,i am paimon！.wav
├── ja
│   ├── Graha-Finetuned
│   │   └── long
│   │       └── 今日の天気もいいですね......一緒に出かけますか.wav
│   ├── Hythloadeus-Finetuned
│   │   └── long
│   │       └── 今日の天気もいいですね。一緒に出かけますか.wav
│   ├── Ryne-Finetuned
│   │   └── long
│   │       └── 今日の天気もいいですね。一緒に出かけますか.wav
│   └── Venat-Finetuned
│       └── long
│           └── 今日の天気もいいですね......一緒に出かけますか.wav
└── zh
    └── bronya
        └── 重装小兔19C，出击！.wav

13 directories, 7 files
```

其中二级子目录表示的是VALL-E-X生成的语言，而三级目录是微调的角色名称，四级子目录代表其输入的模型是否为长短模式。

使用如下命令启动服务：

```bash
nohup uvicorn HTTPServer:app --reload > ./logs/server.log &
```

### 接口使用

```http
POST /generate
```

##### 请求体

| 参数名称       | 类型     | 定义                                |
| ---------- | ------ | --------------------------------- |
| textPrompt | string | 指定VALL-E-X模型将生成的文本提示词             |
| character  | string | 指定VALL-E-X将从presets目录下使用哪个预设来生成声音 |
| language   | string | 指定VALL-E-X模型生成的语言                 |
| noaccent   | bool   | 指定VALL-E-X模型是否按地区口音进行生成           |

##### 响应体

| 响应名称     | 类型     | 用义                |
| -------- | ------ | ----------------- |
| audioURL | string | 缓存音频在PHP服务器上的映射路径 |
| code     | int    | 指定服务器处理的响应码       |
| error    | string | 当服务发生内部错误时返回的信息   |

##### 用例-1：请求了一个不存在于预制中的角色

###### 请求体

```json
{
    "textPrompt":"今日の天気もいいですね。一緒に出かけますか。",
    "character":"ElinLiu",
    "language":"ja", // 由于目前preset中的角色均为日文角色，因此建议使用ja参数
    "noaccent":false
}
```

###### 响应体

```json
{
    "error": "Character not found",
    "code": "404",
    "audioURL": null
}
```

##### 用例-2：常规请求

###### 请求体

```json
{
    "textPrompt":"今日の天気もいいですね。一緒に出かけますか。",
    "character":"Ryne-Finetuned",
    "language":"ja", 
    "noaccent":false,
    "longprompt":false
}
```

###### 响应体

```json
{
    "audioURL": "http://localhost:8080/ja/Ryne-Finetuned/今日の天気もいいですね。一緒に出かけますか。.wav",
    "code": "200",
    "error": null
}
```

##### 用例-3：长文本请求

###### 请求体

```json
{
    "textPrompt":"今日の天気もいいですね......一緒に出かけますか",
    "character":"Graha-Finetuned",
    "language":"ja",
    "noaccent":false,
    "longprompt":true
}
```

###### 响应体

```json
{
    "audioURL": "http://localhost:8080/ja/long/Graha-Finetuned/今日の天気もいいですね......一緒に出かけますか.wav",
    "code": "200",
    "error": null
}
```

### 目前可用的声音角色

由于VALL-E-X原仓库作者并未声明角色的原始语料语言，因此极度不建议使用！产生的任何抽象发音无法进行解释。

| 角色名称                           | 语言     | 表现度                                      | 下载地址                                                                                                                         |
| ------------------------------ | ------ | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Venat-Finetuned（维涅斯）           | ja（日语） | 尚可，极偶尔情况会出现离调现象                          | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Venat-Finetuned.npz          |
| Ryne-Finetuned（琳）              | ja（日语） | 目前听起来最像原作的模型                             | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Ryne-Finetuned.npz           |
| Hythloadeus-Finetuned（希斯拉德）    | ja（日语） | 尚可，并未进行大量测试                              | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Hythloadeus-Finetuned.npz    |
| Graha-Finetuned（小红猫）           | ja（日语） | 使用全新的语料进行微调后，之前的异常情况得到显著改善               | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Graha-Finetuned.npz          |
| Haurchefant-Finetuned（奥尔什方）    | ja（日语） | 尚可                                       | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Haurchefant-Finetuned.npz    |
| MasayoshiSoken-Finetuned（祖坚正庆） | ja（日语） | 尚可，偶尔会出现语塞的情况                            | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/MasayoshiSoken-Finetuned.npz |
| Y'shtola-Finetuned（雅.修特拉）      | ja（日语） | 使用全新的语料进行微调后，之前的异常情况得到显著改善               | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Y'shtola-Finetuned.npz       |
| Krile-Finetuned（可露儿）           | ja（日语） | 效果同琳的微调一样，接近原始声优音色                       | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Krile-Finetuned.npz          |
| Alisaie-Finetuned（阿莉塞）         | ja（日语） | 效果同琳的微调一样，接近原始声优音色                       | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Alisaie-Finetuned.npz        |
| Thancred-Finetuned（桑克瑞德）       | ja（日语） | 声音质量没什么问题，但是模型在情感处理上显得像是声优本人在说话，而非带入角色当中 | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Thancred-Finetuned.npz       |
| Estinien-Finetuned（埃斯蒂尼安）      | ja（日语） | 问题同上                                     | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Estinien-Finetuned.npz       |
| Varshahn-Finetuned（瓦尔桑）        | ja（日语） | 尚未经过大量提示词测试                              | https://raw.githubusercontent.com/ElinLiu0/VALLEX-Inference-Http/master/Finetuned-VALL-E_Prompt/Varshahn-Finetuned.npz       |

<style></style>

### 可添加的优化选项

● requests.exceptions.ProxyError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /charactr/vocos-encodec-24khz/resolve/main/config.yaml (Caused by ProxyError('Cannot connect to proxy.', TimeoutError('_ssl.c:975: The handshake operation timed out')))")

> 该问题的解决方法相对来说简单粗暴：在网络允许的前提下将模型进行克隆：
> 
> ```bash
> ```
> git clone https://huggingface.co/charactr/vocos-encodec-24khz
> ```
> ```

> 需要说明：执行该克隆前请确保git-lfs被正确安装后方可执行，本地克隆好后上传到服务器VALL-E-X路径下。并在utils/generation.py下的preload_models()函数，约90-91行，修改为：
> 
> ```python
> ```
> vocos = Vocos.from_pretrained('./charactr/vocos-encodec-24khz').to(device)
> ```
> ```

> 由于添加了./的缘故，因此huggingface会跳过缓存与远程库检查，取而代之从本地加载。

<style></style>

● 可用的约1~1.2倍性能提升

> 同样是打开utils/generation.py，preload_models()函数下，将位于约82~84行处添加一行代码：
> 
> ```python
> model = torch.compile(model,backend="cudagraphs" if torch.cuda.is_available() else "onnxrt",mode="reduce-overhead" if torch.cuda.is_available() else "default")
> ```

> 这么做的目的是将模型在程序内部进行编译为CUDA图模型或者ONNX图模型，可充分利用GPU和CPU的计算资源。
> 
> 与之相同的操作，在vocos变量下添加：
> 
> ```bash
> vocos = torch.compile(vocos,backend="cudagraphs" if torch.cuda.is_available() else "onnxrt",mode="reduce-overhead" if torch.cuda.is_available() else "default")
> ```

> 经测试，在RTX3080Ti 16GB GPU上：未经`torch.compile()`进行动态图编译优化时，但语料的推理时长约为2.5~3秒，经过`torch.compile()`动态图编译优化后，平均推理时长约为1.2秒，部分较短的文本提示词下甚至可以达到1秒以下。
> 
> 请注意，为了防止使用ONNXRT进行动态图优化时产生ONNX依赖报错，请使用下述命令安装ONNX：
> 
> ```bash
> pip install onnx onnxruntime onnxsim
> #当安装CUDA GPU时请将onnxruntime替代成onnxruntime-gpu
> ```

### 关于结束进程后仍在工作的情况

由于kill指令不会去自动kill掉由subprocess拉起的子进程，因此需要手动通过ps获取php-cli的进程ID：

```bash
ps aux | grep php
```

定位到进程ID后手动使用kill来关闭：

```bash
kill -9 <PHP_CLI_PID>
```

同时针对fastAPI仍然处于工作状态的情况，请手动关闭如下进程：

```bash
python -c from multiprocessing.resource_tracker import main;main(4)
python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7) --multiprocessing-fork
```

### 多卡推理

暂无，在搜。且并无太大必要，推理的压力比训练计算要小太多。

### 安全性更新
更新了一项API防劫持的措施，即：静态时间戳验证算法。  
该算法的实现方式是：通过从客户端的请求字段中获取请求发起时的时间，并在算法内得到带偏移量服务器的响应时间。将来自客户端的请求时间分别与响应时间和偏移时间轴进行比较以验证该请求的合法性，详情见`ShiftedTimeStampValidator.py`。

### TODO

- [ * ] 实现基于位移量的时间戳验证算法，以降低API被劫持的风险。
