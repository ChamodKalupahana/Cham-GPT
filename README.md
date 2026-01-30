# Cham-GPT

This is the revial of the project I started a couple of years ago where I tried to build a chatbot of myself but didn't know enough about transformers so I built an RNN which is in `RRN-model/`. Now after doing ARBOx, I do know enough know transformers to build them from stratch! I'm taking this project on as a challenge, writing as much of the code myself with as little guidance as possible so that I'll be prepared when I have to replicate a much more novel model in the future.

Plan:
- Build GPT-2 from stratch
- Base Cham-GPT on GPT-2 arch and train on my own data
- Interpret Cham-GPT, ideally for a direction of "Cham-ness"
- Deploy publicly on HuggingFace


## Setup

```python
conda create -n cham-gpt python=3.11 -y
conda activate cham-gpt
pip install -r requirement.txt
```