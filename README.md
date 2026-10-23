# Tool calling from scratch

An explainer for clients on how LLMs "call tools": the model only writes text,
and a normal program runs the tool. Every piece is built by hand:

- **The model** - Gemma 3 270M in plain PyTorch, taken from my
  [lets-build-llms](https://github.com/jSwords91/lets-build-llms) repo, with the
  weights of FunctionGemma (Google's version of it, trained for function calling).
- **The prompt** - the conversation and the tools written out as one string, so
  you can read exactly what the model sees.
- **The parser** - generation stops at a special token, and the tool call gets
  pulled out of the text.
- **The loop** - the program runs the function, adds the result, and hands back
  to the model to finish its answer.

## Quickstart

You'll need [uv](https://docs.astral.sh/uv/) and a Hugging Face account.
FunctionGemma is gated, so first accept the licence on its
[model page](https://huggingface.co/google/functiongemma-270m-it), then log in
with `uv run hf auth login` (or set `HF_TOKEN`).

```sh
uv sync
uv run python tool_calling.py
```

The first run downloads about 512MB of weights. It'll use your GPU if you have
one, otherwise the CPU (which is a lot slower).

It prints the prompt the model starts with, then waits for you to type. Try
`What's the weather in London?` or `What files are in /tmp?`. It's only a 270M
model, so keep requests simple and paths short. (`get_weather` just returns
made-up weather, but `list_files` is real.)

## What actually happens

Here's the full text after asking about the weather. The response line is
written by the program (it shows up cyan in the terminal), everything after
`<start_of_turn>model` apart from that is written by the model:

```
<start_of_turn>user
What's the weather in London?<end_of_turn>
<start_of_turn>model
<start_function_call>call:get_weather{city:<escape>London<escape>}<end_function_call>
<start_function_response>response:get_weather{value:<escape>15C and light rain in London<escape>}<end_function_response>
The current weather in London is 15°C with light rain in London.<end_of_turn>
```

1. The model writes `call:get_weather{...}`, then the token
   `<start_function_response>`, and stops. That token is basically the model
   saying "run this, I'll wait".
2. The program reads the call out of the text, runs `get_weather("London")`,
   and writes the result back in the same format.
3. The model runs again on the longer text and finishes its answer, now that it
   knows the weather.

So that's all "calling a tool" really is: the model writes some text, and a
program acts on it. (I've added line breaks around the call and the response to
make it easier to read. In the real text they run on.)

## Reading guide

The git history is the tutorial. Each commit adds one idea:

1. **README** - this file.
2. **model** - Gemma 3 from scratch, chatting with no tools.
3. **prompt** - tools are just text in the prompt.
4. **parse** - a tool call is just tokens the model writes.
5. **loop** - the program runs the tool and hands control back.

Read them in order with `git log --reverse -p`.
