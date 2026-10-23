# Tool calling from scratch

A language model only ever predicts the next token. So how does it "call a
tool"?

It doesn't. The tools are written into the prompt as text, the model writes a
tool call as text, and a plain program reads that text, runs the function, and
writes the result back into the prompt. This repo builds every piece of that by
hand, with a small model we can see all the way through:

- **The model** - Gemma 3 270M in plain PyTorch, loaded with the weights of
  FunctionGemma, Google's version of it trained for function calling.
- **The prompt** - the conversation and the tool definitions turned into one
  string, by hand, so you can read exactly what the model sees.
- **The parser** - generation stops at a special token, and the tool call is
  pulled out of the text.
- **The loop** - the program runs the function, appends the result, and hands
  control back to the model to finish its answer.

For pedagogical purposes only.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a Hugging Face account.
FunctionGemma is gated: accept the licence on its
[model page](https://huggingface.co/google/functiongemma-270m-it), then log in
with `uv run hf auth login` (or set `HF_TOKEN`).

```sh
uv sync
uv run python tool_calling.py
```

The first run downloads about 512MB of weights. It uses your GPU if there is
one, otherwise the CPU.

## Reading guide

The git history is the tutorial. Each commit adds one idea:

1. **README** - this file.
2. **model** - Gemma 3 from scratch, chatting with no tools.
3. **prompt** - tools are just text in the prompt.
4. **parse** - a tool call is just tokens the model writes.
5. **loop** - the program runs the tool and hands control back.

Read them in order with `git log --reverse -p`.
