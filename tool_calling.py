"""Tool calling from scratch: a small Gemma, a hand-written prompt, a parser and a loop."""
import os

# Every generation step re-reads a slightly longer sequence, so GPU buffers grow
# a little each token. Without this, PyTorch's cache keeps reserving fresh blocks
# for them until the GPU runs out, even though little is actually in use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from gemma3 import Gemma3Model, GemmaTokenizer, ModelConfig, load_pretrained_weights

REPO = "google/functiongemma-270m-it"  # Gemma 3 270M, fine-tuned by Google for function calling
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    """Build the model from scratch, then pour FunctionGemma's weights into it."""
    model = Gemma3Model(ModelConfig())
    load_pretrained_weights(model, load_file(hf_hub_download(REPO, "model.safetensors")))
    tokenizer = GemmaTokenizer(hf_hub_download(REPO, "tokenizer.json"))
    return model.to(DEVICE).eval(), tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Greedy decoding: run the whole sequence through the model, append the likeliest
    next token, repeat until <end_of_turn>. There is no KV cache, so every step
    re-reads the whole sequence: slow, but nothing is hidden."""
    ids = tokenizer.encode(prompt)
    stop = tokenizer.tokenizer.token_to_id("<end_of_turn>")
    new = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids + new], device=DEVICE))
        next_id = logits[0, -1].argmax().item()
        if next_id == stop:
            break
        new.append(next_id)
    return tokenizer.decode(new)


def main():
    model, tokenizer = load()
    while True:
        try:
            user = input("> ")
        except (EOFError, KeyboardInterrupt):
            return
        print(generate(model, tokenizer, GemmaTokenizer.apply_chat_template(user)))


if __name__ == "__main__":
    main()
