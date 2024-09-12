# MediMate
# How to run?
### STEPS:
Clone the repository
```bash
Project repo: https://github.com/mehul-1999/MediMate

```
### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n mchat python=3.8 -y
```
```bash
conda activate mchat
```
### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```
### Create a '.env' file in the root directory and add your Pinecone
credentials as follows:
```ini
PINECONE_API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
PINECONE_API_ENV = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
...
### Download the quantize model from the link provided in model folder &    
keep the model in the model directory:
```ini
## Download the Llama 2 Model:
llama-2-7b-chat.Q5_K_M.gguf
## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
```
