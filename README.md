# Charlie
Charlie is an uncensored LLM chatbot based on the Solar-10.7B model fine-tuned with the toxic-dpo-v0.1 dataset to remove censorship and alignment.

# Uncensored-LLM-Chatbot
<a href="https://huggingface.co/w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored">SOLAR-10.7B-Instruct-v1.0-uncensored</a> was created by fine-tuning the <a href="https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0">base Solar-10.7B model</a> with the <a href="https://huggingface.co/datasets/unalignment/toxic-dpo-v0.1">toxic-dpo-v0.1 dataset</a> to remove censorship and alignment.

# Usage

`panel serve app.py`

When run for the first time, it will first download the model file, which might take a couple of minutes.
