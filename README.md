# Cham-GPT

## 29/1/26
This is the revival of the project I started a couple of years ago where I tried to build a chatbot of myself but didn't know enough about transformers so I built an RNN which is in `RNN-model/`. Now after doing ARBOx, I do know enough about transformers to build them from scratch! I'm taking this project on as a challenge, writing as much of the code myself with as little guidance as possible so that I'll be prepared when I have to replicate a much more novel model in the future.

Plan:
- Build GPT-2 from scratch
- Base Cham-GPT on GPT-2 arch and train on my own data
- Interpret Cham-GPT, ideally for a direction of "Cham-ness"
- Deploy publicly on HuggingFace
- Host and allow public to inference the model at [chamgpt.net](https://chamgpt.net/)


## Setup

```bash
conda create -n cham-gpt python=3.11 -y
conda activate cham-gpt
pip install -r requirements.txt
```