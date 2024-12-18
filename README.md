# Gemini Multimodal Live API command-line client

This is a full-featured client for the Gemini Multimodal Live API, written in Python with minimal dependencies.

**Please note that you should use headphones with this client. There is no echo cancellation.**

The client supports:
  - text, audio, and screen capture video input
  - text or audio output
  - setting the system instruction
  - setting an initial message with a command-line arg
  - the grounded search built-in tool
  - the code execution built-in tool
  - importing functions from a file and automatically generating Gemini function declarations

This code uses the Multimodal Live WebSocket API directly. It was originally written as a standalone testbed to experiment with the API. In its current form, it's useful for testing API features and for sharing bits of code with other people who are developing with Gemini 2.0.

The Multimodal Live API is a developer preview and is evolving rapidly. We'll try to update the implementation and docs here frequently!

## Installation

```
# Tested with Python 3.12 and venv, but other versions and virtual environments should work
python3.12 -m venv venv
source venv/bin/activate

# install with all dependencies
pip install pyaudio websockets google-genai mss

# or with only basic dependencies (no function calling or screenshare support)
pip install pyaudio websockets
```

## Usage

To start a voice-to-voice session:

```
source venv/bin/activate
EXPORT GOOGLE_API_KEY=...
python gemini-live.py
```

Command-line arguments are described in the `--help` message.

Boolean arguments follow the traditional command-line boolean flag format, e.g. `--audio-input` to enable audio input, and `--no-audio-input` to disable audio input.
 
```
% python gemini-live.py --help
usage: gemini-live.py [-h] [--system-instruction SYSTEM_INSTRUCTION] [--initial-message INITIAL_MESSAGE]
                      [--initial-message-delay INITIAL_MESSAGE_DELAY] [--voice VOICE] [--audio-input | --no-audio-input]
                      [--audio-output | --no-audio-output] [--text-output | --no-text-output] [--search | --no-search]
                      [--code-execution | --no-code-execution] [--screen-capture-fps SCREEN_CAPTURE_FPS]
                      [--import-functions IMPORT_FUNCTIONS]

Gemini Multimodal Live API client

options:
  -h, --help            show this help message and exit
  --system-instruction SYSTEM_INSTRUCTION
                        System instruction text
  --initial-message INITIAL_MESSAGE
                        First 'user' message to send to the model
  --initial-message-delay INITIAL_MESSAGE_DELAY
                        Delay in seconds before sending the initial message
  --voice VOICE         Voice name. Options are Puck, Charon, Kore, Fenrir, and Aoede
  --audio-input, --no-audio-input
                        Enable/disable audio input. Default is enabled.
  --audio-output, --no-audio-output
                        Enable/disable audio output. Default is enabled.
  --text-output, --no-text-output
                        Enable/disable text output. Default is disabled. Audio output and text output cannot be enabled
                        at the same time. Enabling text output will disable audio output.
  --search, --no-search
                        Enable/disable the built-in grounded search tool.
  --code-execution, --no-code-execution
                        Enable/disable the built-in code execution tool.
  --screen-capture-fps SCREEN_CAPTURE_FPS
                        Enable screen capture. Specify a frames-per-second value. For example, 1.0 for one frame per
                        second.
  --import-functions IMPORT_FUNCTIONS
                        Import functions from a Python file. Specify a filename.
```

## Examples

Set a system instruction and start the conversation by sending an initial 'user' message to Gemini.

```
python gemini-live.py --system-instruction "Talk like a pirate." --initial-message "What's the origin of the phrase, the wine-dark sea?"
```

The voice is set to Charon by default. Set the voice to Aoede.

```
python gemini-live.py --voice Aoede
```

Disable audio input. (Note: you can type to send messages to Gemini whether audio input is enabled or not. Enter (a new line) sends whatever text you've typed. No terminal text, history, or escape key management is implemented. This is very bare-bones text input!)

```
python gemini-live.py --no-audio-input --initial-message 'Hello!'
```

Switch from audio to text output. Currently you have to choose between audio and text output. Both modes are not supported at the same time.

```
python gemini-live.py --text-output --initial-message 'Say the alphabet from A-Z'
```

Enable the built-in search tool.

```
python gemini-live.py --search --initial-message 'Look up the current season-to-date snowpack percentage in the central Sierra Nevada mountains.'
```

Enable code execution.

```
python gemini-live.py --text-output --code-execution --initial-message 'Make up a sentence about each of these four fruits: strawberry, apple, banana, blueberry. Tell me each sentence. Then use code execution to sort the sentences by length and say each sentence again in length-sorted order.'
```

Enable video screen capture. Note that only audio-triggered inference will be able to access the video stream input. Text-triggered inference will either hallucinate or repeat previous information.

```
python gemini-live.py --screen-capture-fps 2.0 
```

Import functions from a file and set the API up to use them. The code for this makes use of the very nice function declaration auto-generation in the new `google-genai` SDK.

```
python gemini-live.py --import-functions function-examples.py --initial-message "print the word 'hello' to the console."
```

## Code walk-through

The full [gemini-live.py](gemini-live.py) file is about 480 lines, but we'll skip the imports and argument parsing code.

That leaves about 320 lines of code that implement all of the above features except for function calling. The function declaration and calling code is in a separate module, [function_helper.py](function_helper.py), which is about 100 lines (most of it docstrings).

### `__main__` and the run loop

This program needs to perform several long-running tasks:

  - receive WebSocket messages
  - send audio frames for playout
  - read audio frames from the OS and send them out via the WebSocket
  - read text from stdin and send lines of text out via the WebSocket
  - if screen-capture video is enabled, capture frames and send them out via the WebSocket

There are a number of ways to design this in Python. This program uses threads for OS audio input, OS audio playout, and reading text from stdin. For everything else, asyncio tasks are used.

```python
class AudioStreamer:
    def __init__(self):
        self.running = False
        self.event_loop = None
        self.mic_audio_in = None
        self.speaker_audio_out = None
        self.speaker_audio_buffer = bytearray()
        self.p = pyaudio.PyAudio()

    async def run(self):
        self.event_loop = asyncio.get_event_loop()
        self.running = True

        if AUDIO_INPUT:
            self.mic_audio_in = self.p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=MIC_SAMPLE_RATE,
                input=True,
                stream_callback=self.mic_audio_in_callback,
                frames_per_buffer=int(MIC_SAMPLE_RATE / 1000) * 2 * 50,  # 50ms (S16_LE is 2 bytes)
                start=False,
            )

        self.speaker_audio_out = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SPEAKER_SAMPLE_RATE,
            output=True,
            frames_per_buffer=256,
            stream_callback=self.speaker_audio_out_callback,
        )

        try:
            self.ws = await websockets.connect(
                uri=f'wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={os.getenv("GOOGLE_API_KEY")}'
            )
            print("Connected to Gemini")
        except Exception as e:
            print(f"Exception: {e}")
            return

        asyncio.create_task(self.stdin_worker())
        asyncio.create_task(self.ws_receive_worker())
        asyncio.create_task(self.print_audio_output_buffer_info())
        if SCREEN_CAPTURE_FPS:
            asyncio.create_task(self.screen_capture_worker())

        try:
            await self.setup_model()
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Exception: {e}")
        finally:
            print("Exiting ...")
            self.running = False
            sys.stdin.close()
            self.p.terminate()
            await self.ws.close()


if __name__ == "__main__":
    parse_args()
    asyncio.run(AudioStreamer().run())
```

The `run()` function sets everything up and then loops until either ctrl-c or some piece of code sets `self.running` to false.

### The `setup` message — sending a configuration to the API

The first message we send after opening the WebSocket connection configures various features of the API.

```python
    async def setup_model(self):
        try:
            response_modality = []

            # Currently the API will throw an error if both response modalities are requested. This
            # will likely change soon, though.
            if AUDIO_OUTPUT:
                response_modality.append("AUDIO")
            if TEXT_OUTPUT:
                response_modality.append("TEXT")

            setup = {
                "setup": {
                    "model": MODEL,
                    "generation_config": {
                        "response_modalities": response_modality,
                        "speech_config": {
                            "voice_config": {"prebuilt_voice_config": {"voice_name": VOICE}},
                        },
                    },
                    "tools": [],
                },
            }
            if SYSTEM_INSTRUCTION_TEXT:
                print("System instruction enabled")
                setup["setup"]["system_instruction"] = {
                    "parts": [
                        {
                            "text": SYSTEM_INSTRUCTION_TEXT,
                        }
                    ]
                }
            if SEARCH:
                print("Search enabled")
                setup["setup"]["tools"].append({"google_search": {}})
            if CODE_EXECUTION:
                print("Code execution enabled")
                setup["setup"]["tools"].append({"code_execution": {}})
            if FUNCTION_DECLARATIONS:
                setup["setup"]["tools"].append({"function_declarations": FUNCTION_DECLARATIONS})
            print("Sending setup", setup)
            await self.ws.send(json.dumps(setup))
        except Exception as e:
            print(f"Exception: {e}")
```

### Receiving events from the WebSocket

To receive audio output, text output, and events, we need to read from the WebSocket.

```python
    async def ws_receive_worker(self):
        try:
            async for m in self.ws:
                if not self.running:
                    break
                evt = json.loads(m)
                self.print_evt(evt, m)

                if evt.get("setupComplete", None) is not None:
                    asyncio.create_task(self.send_initial_message())
                    print("Ready: say something to Gemini")
                    if self.mic_audio_in:
                        self.mic_audio_in.start_stream()
                elif sc := evt.get("serverContent"):
                    if sc.get("interrupted"):
                        print("Interrupted by server")
                        self.speaker_audio_buffer.clear()
                        continue
                    if parts := sc.get("modelTurn", {}).get("parts"):
                        if text := parts[0].get("text"):
                            print(f"  <- {text}")
                        elif inline_data := parts[0].get("inlineData"):
                            mime_str = inline_data.get("mimeType")
                            mime_type, sample_rate = re.match(
                                r"([\w/]+);rate=(\d+)", mime_str
                            ).groups()
                            if mime_type == "audio/pcm" and sample_rate == str(SPEAKER_SAMPLE_RATE):
                                audio = base64.b64decode(inline_data.get("data"))
                                self.speaker_audio_buffer.extend(audio)
                            else:
                                print(f"Unsupported mime type or sample rate: {mime_str}")
                        if code := parts[0].get("executableCode"):
                            pass
                elif tool_call := evt.get("toolCall"):
                    await self.handle_tool_call(tool_call)
        except Exception as e:
            print(f"Exception: {e}")
            self.running = False
```

We recognize 3 types of messages:

- setupComplete
- ServerContent
- toolCall

The Multimodal Live API transmits errors as WebSocket status codes (with messages where applicable). Generally, an error will close the connection. When this happens, we set `self.running` to False to being the process of exiting our run loop.

The serverContent messages can:
- contain audio or text output from the model
- indicate that the API has detected user speech and `interrupted` the current output
- contain information about code execution

### Audio

The PyAudio `open()` calls set up mic audio input and speaker audio output in callback mode, which uses threads internally. We request 256 `frames_per_buffer` in both cases, which is a reasonable balance between latency and robustness to stalling audio input/output. (Even when using callback mode, activity on the Python main thread can impact audio performance.)

Here are our audio callback functions.

```python
    def mic_audio_in_callback(self, in_data, frame_count, time_info, status):
        if not self.running:
            return (None, pyaudio.paComplete)
        self.event_loop.create_task(self.send_audio(in_data))
        return (None, pyaudio.paContinue)

    def speaker_audio_out_callback(self, in_data, frame_count, time_info, status):
        if not self.running:
            return (bytes(frame_count * CHANNELS * 2), pyaudio.paComplete)
        audio = bytes(self.speaker_audio_buffer[: frame_count * CHANNELS * 2])
        del self.speaker_audio_buffer[: frame_count * CHANNELS * 2]
        audio += b"\0" * (frame_count * CHANNELS * 2 - len(audio))
        return (audio, pyaudio.paContinue)
```

### Receiving audio

When we receive audio from the WebSocket, we base64 decode it and put it in a buffer that the `speaker_audio_out_callback` above will read from on-demand. 

```python
                        elif inline_data := parts[0].get("inlineData"):
                            mime_str = inline_data.get("mimeType")
                            mime_type, sample_rate = re.match(
                                r"([\w/]+);rate=(\d+)", mime_str
                            ).groups()
                            if mime_type == "audio/pcm" and sample_rate == str(SPEAKER_SAMPLE_RATE):
                                audio = base64.b64decode(inline_data.get("data"))
                                self.speaker_audio_buffer.extend(audio)
                            else:
                                print(f"Unsupported mime type or sample rate: {mime_str}")
```

### Sending audio 

When we receive audio from the mic, we call our `send_audio()` function to send it out the WebSocket as soon as possible. We want to avoid issues with trying to use the WebSocket connection from multiple threads. So we create a task on the main thread using `self.event_loop.create_task()` to run this function.

Here's `send_audio()`. We're just base64 encoding the raw audio bytes and sending them to Gemini as a `realtimeInput` message.

```python
    async def send_audio(self, raw_audio):
        payload = base64.b64encode(raw_audio).decode("utf-8")
        try:
            msg = json.dumps(
                {
                    "realtimeInput": {
                        "mediaChunks": [
                            {
                                "mimeType": f"audio/pcm;rate={MIC_SAMPLE_RATE}",
                                "data": payload,
                            }
                        ],
                    },
                }
            )
            await self.ws.send(msg)
        except Exception as e:
            print(f"Exception: {e}")
            self.running = False
```

### Receiving text

When we receive text output from the model, we just print it out.

```python
                         if text := parts[0].get("text"):
                            print(f"  <- {text}")
```

### Sending text

To send text to the model, we create and send a clientContent message. In this program, we always set `turnComplete` to true in this message. If we set `turnComplete` to false, the model will wait to perform inference until a clientContent message with `turnComplete` set to true is received.

Note that clientContent messages are used to send text, and realtimeInput messages are used to send audio and video.

```python
    async def send_text(self, text):
        try:
            print(f"  -> {text}")
            await self.ws.send(
                json.dumps(
                    {
                        "clientContent": {
                            "turns": [
                                {"parts": [{"text": text}], "role": "user"},
                            ],
                            "turnComplete": True,
                        }
                    }
                )
            )
        except Exception as e:
            print(f"Exception: {e}")
            self.running = False
```

### Sending video

Sending video is almost exactly like sending audio. Video frames are jpeg byte strings, base64 encoded.

```python
    async def send_video(self, jpg_bytes):
        payload = base64.b64encode(jpg_bytes).decode("utf-8")
        try:
            msg = json.dumps(
                {
                    "realtimeInput": {
                        "mediaChunks": [
                            {
                                "mimeType": "image/jpg",
                                "data": payload,
                            }
                        ],
                    },
                }
            )
            await self.ws.send(msg)
        except Exception as e:
            print(f"Exception: {e}")
            self.running = False
```

### Built-in tools: search and code execution

The operations of the built-in search and code execution tools are entirely performed on the server end of the API. Information about them is included in serverContent messages in `groundingMetadata` and `executableCode` parts.

See documentation about groundingMetadata here:
* https://ai.google.dev/gemini-api/docs/grounding?lang=rest

We print out the url of each `groundingChunk` in our `print_evt()` function.

```
    def print_evt(self, evt, response):
        columns, rows = shutil.get_terminal_size()
        maxl = columns - 5
        print(str(evt)[:maxl] + " ...")
        if grounding := evt.get("serverContent", {}).get("groundingMetadata"):
            for chunk in grounding.get("groundingChunks", []):
                print(f"  <- {chunk.get("web").get("title")}")
        # ...
```

Similarly, see documentation for `executableCode` here:
* https://ai.google.dev/gemini-api/docs/code-execution?lang=rest

Note that currently the `executableCode` content parts are not sent over the WebSocket when the output modality is audio.

### Function calling

For the API to call user-provided functions, you need to:
1. Provide function declarations in the `tools[]` list passed to the object as part of the setup message at the beginning of the connection.
2. Call your function when you receive a toolCall message from the LLM.
3. Send back a toolResponse message with the data produced by your function.

It's worth noting that because the LLM's request for a function call is completely decoupled from your execution of the function, you have wide latitude in how you actually perform function calls.

The function declarations you provide to the LLM are descriptions of notional functions that don't actually need to exist. You could, for example, implement a toolCall handler that:
  - calls a remote function using an HTTP request
  - looks information up in a database, using the function name and arguments to create a SQL query
  - mocks a function for testing, returning static data that can be used as part of an test/evals suite

The most common setup for user-defined functions, though, is to actually call a locally defined function that does exist!

Writing function declarations by hand is a little bit laborious. Here's the canonical `get_current_weather` function declaration that's used in lots of docs and examples.

```
tools = [
    {
        "function_declarations": [
            {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        ]
    }
]
```

The `google-genai` SDK can automatically generate function declarations from a combination of Python introspection and docstrings. 

If you have an installed copy of that SDK, you can read the code for this. It's in venv/lib/python3.12/site-packages/google/genai/types.py :: `from_function()`

We've written a wrapper for that code that auto-generates function declarations for all functions declared in a file. That wrapper is [function_helpers.py] and is imported if you specify the `--import-functions` command-line option.

Note that the LLM's ability to call your function properly will depend heavily on the quality of the docstring you write for the function.

You can test with or use as a starting point, the [function-examples.py] file.

```
python gemini-live.py --import-functions function-examples.py --initial-message "print the word 'hello' to the console."
```

To send function call results back to the API,
- create a toolResponse message.
- include the `id` and the `name` of the function that the API provided in the toolCall message.
- put the function call result in the `response` field of the message.

```python
    async def handle_tool_call(self, tool_call):
        responses = []
        for f in tool_call.get("functionCalls", []):
            print(f"  <- Function call: {f}")
            response = await function_helper.call_function(
                FUNCTION_IMPORTS_MODULE, f.get("name"), **f.get("args")
            )
            responses.append(
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "response": response,
                }
            )
        msg = json.dumps(
            {
                "toolResponse": {
                    "functionResponses": responses,
                }
            }
        )
        print(f"  -> {msg}")
        await self.ws.send(msg)
```

## Other resources

* Multimodal Live [API overview](https://ai.google.dev/api/multimodal-live)
* Gemini 2.0 [cookbook](https://github.com/google-gemini/cookbook/tree/main/gemini-2)
* A [starter app](https://github.com/google-gemini/multimodal-live-api-web-console) from the Google Creative Labs team
* Multimodal Live [WebRTC client app starter kit](https://github.com/pipecat-ai/gemini-multimodal-live-demo)
