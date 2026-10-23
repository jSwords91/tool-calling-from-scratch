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

# Tools are described the way most APIs take them: a name, a description, and a
# JSON schema for the arguments. The model never sees these dicts, only the text
# that declare() turns them into.
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name."}},
            "required": ["city"],
        },
    },
    {
        "name": "list_files",
        "description": "List the files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The directory to list."}},
            "required": ["path"],
        },
    },
]


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


def declare(tool):
    """Write a tool definition as the text FunctionGemma was trained to read.
    <escape> wraps every string, so quotes and braces inside it are not syntax.
    (Handles string parameters only, which is all our tools need.)"""
    def text(s):
        return f"<escape>{s}<escape>"
    params = tool["parameters"]
    properties = ",".join(
        f"{name}:{{description:{text(p['description'])},type:{text(p['type'].upper())}}}"
        for name, p in sorted(params["properties"].items())
    )
    required = ",".join(text(name) for name in params["required"])
    return (
        f"<start_function_declaration>declaration:{tool['name']}{{description:{text(tool['description'])},"
        f"parameters:{{properties:{{{properties}}},required:[{required}],type:{text('OBJECT')}}}}}"
        "<end_function_declaration>"
    )


def system_turn(tools):
    """The developer turn: an instruction, then every tool declared as text."""
    instruction = "You are a model that can do function calling with the following functions"
    return f"<start_of_turn>developer\n{instruction}{''.join(declare(t) for t in tools)}<end_of_turn>\n"


def main():
    model, tokenizer = load()
    prompt = system_turn(TOOLS)  # the whole conversation is this one string
    print(prompt)  # this is everything the model knows about its tools
    while True:
        try:
            user = input("> ")
        except (EOFError, KeyboardInterrupt):
            return
        prompt += f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        reply = generate(model, tokenizer, prompt)
        print(reply)
        prompt += reply + "<end_of_turn>\n"


if __name__ == "__main__":
    main()
