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
one, otherwise the CPU (much slower).

It prints the prompt the model starts from, then waits for you. Try
`What's the weather in London?` or `What files are in /tmp?`. It is a 270M
model, so keep requests simple and paths short.

## What actually happens

Ask about the weather and this is the text, token for token, after the
program's work (cyan in the terminal) is added:

```
<start_of_turn>user
What's the weather in London?<end_of_turn>
<start_of_turn>model
<start_function_call>call:get_weather{city:<escape>London<escape>}<end_function_call>
<start_function_response>response:get_weather{value:<escape>15C and light rain in London<escape>}<end_function_response>
The current weather in London is 15°C with light rain.<end_of_turn>
```

1. The model writes `call:get_weather{...}`, then the token
   `<start_function_response>`, and stops. That token is its way of saying
   "run this, I'll wait".
2. The program reads the call out of the text, runs `get_weather("London")`,
   and writes the result in the same format.
3. The model is run again on the longer text and finishes its answer, now
   knowing the weather.

That is all "calling a tool" is: the model writes text, a program acts on it.
(The line breaks around the function call and response are added here for
reading; in the real text they run on.)

## Reading guide

The git history is the tutorial. Each commit adds one idea:

1. **README** - this file.
2. **model** - Gemma 3 from scratch, chatting with no tools.
3. **prompt** - tools are just text in the prompt.
4. **parse** - a tool call is just tokens the model writes.
5. **loop** - the program runs the tool and hands control back.

Read them in order with `git log --reverse -p`.
