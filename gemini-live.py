import argparse
import asyncio
import base64
import json
import io
import os
import pyaudio
import re
import shutil
import select
import sys
import traceback
import websockets


MODEL = "models/gemini-2.0-flash-exp"
HOST = "generativelanguage.googleapis.com"

MIC_SAMPLE_RATE = 16000
SPEAKER_SAMPLE_RATE = 24000
FORMAT = "S16_LE"
CHANNELS = 1

#
# Argument parsing
#

SYSTEM_INSTRUCTION_TEXT = ""
INITIAL_MESSAGE = ""
INITIAL_MESSAGE_DELAY = 0.0
VOICE = None
AUDIO_INPUT = None
AUDIO_OUTPUT = None
TEXT_OUTPUT = None
SEARCH = False
CODE_EXECUTION = False
SCREEN_CAPTURE_FPS = 0.0

function_helper = None
FUNCTION_IMPORTS_MODULE = None
FUNCTION_DECLARATIONS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini Multimodal Live API client")
    parser.add_argument("--system-instruction", type=str, help="System instruction text")
    parser.add_argument(
        "--initial-message", type=str, help="First 'user' message to send to the model"
    )
    parser.add_argument(
        "--initial-message-delay",
        type=float,
        default=0.0,
        help="Delay in seconds before sending the initial message",
    )
    parser.add_argument(
        "--voice",
        default="Charon",
        type=str,
        help="Voice name. Options are Puck, Charon, Kore, Fenrir, and Aoede ",
    )
    parser.add_argument(
        "--audio-input",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable audio input. Default is enabled.",
    )
    parser.add_argument(
        "--audio-output",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable audio output. Default is enabled.",
    )
    parser.add_argument(
        "--text-output",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable text output. Default is disabled. Audio output and text output cannot be enabled at the same time. Enabling text output will disable audio output.",
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable the built-in grounded search tool.",
    )
    parser.add_argument(
        "--code-execution",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable the built-in code execution tool.",
    )
    parser.add_argument(
        "--screen-capture-fps",
        type=float,
        default=0.0,
        help="Enable screen capture. Specify a frames-per-second value. For example, 1.0 for one frame per second.",
    )
    parser.add_argument(
        "--import-functions",
        type=str,
        help="Import functions from a Python file. Specify a filename.",
    )

    args = parser.parse_args()

    global \
        SYSTEM_INSTRUCTION_TEXT, \
        INITIAL_MESSAGE, \
        INITIAL_MESSAGE_DELAY, \
        VOICE, \
        AUDIO_INPUT, \
        AUDIO_OUTPUT, \
        TEXT_OUTPUT, \
        SEARCH, \
        CODE_EXECUTION, \
        SCREEN_CAPTURE_FPS, \
        FUNCTION_IMPORTS_MODULE, \
        FUNCTION_DECLARATIONS

    SYSTEM_INSTRUCTION_TEXT = args.system_instruction
    INITIAL_MESSAGE = args.initial_message
    INITIAL_MESSAGE_DELAY = args.initial_message_delay
    VOICE = args.voice
    AUDIO_INPUT = args.audio_input
    AUDIO_OUTPUT = args.audio_output
    TEXT_OUTPUT = args.text_output
    SEARCH = args.search
    CODE_EXECUTION = args.code_execution
    SCREEN_CAPTURE_FPS = args.screen_capture_fps

    if args.text_output and args.audio_output:
        print(
            "Warning: audio output and text output cannot be enabled at the same time. Disabling audio output."
        )
        AUDIO_OUTPUT = False

    if args.screen_capture_fps:
        try:
            global mss, Image
            import mss
            from PIL import Image
        except Exception:
            print("Screen capture requires the mss library. Install with 'pip install mss'")
            quit

    if args.import_functions:
        try:
            # Conditional import, because the function declaration helpers depend on having the
            # google-genai library installed. If you don't need to import functions, you don't
            # need to install that dependency.
            global function_helper
            import function_helper

            function_declarations, module = function_helper.create_function_declarations_from_file(
                args.import_functions
            )
            FUNCTION_IMPORTS_MODULE = module
            FUNCTION_DECLARATIONS = function_declarations
        except Exception:
            print("Function import failed")
            print(traceback.print_exc())
            quit()


#
# Main application class
#


class AudioStreamer:
    def __init__(self):
        self.running = False
        self.event_loop = None
        self.mic_audio_in = None
        self.speaker_audio_out = None
        self.speaker_audio_buffer = bytearray()
        self.p = pyaudio.PyAudio()

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

    async def send_initial_message(self):
        if INITIAL_MESSAGE:
            await asyncio.sleep(INITIAL_MESSAGE_DELAY)
            await self.send_text(INITIAL_MESSAGE)

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

    async def handle_tool_call(self, tool_call):
        # print(f"  <- handling tool call {tool_call}")
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

    async def print_audio_output_buffer_info(self):
        while self.running:
            if self.speaker_audio_buffer:
                print(
                    f"Current audio buffer size: {len(self.speaker_audio_buffer) / (SPEAKER_SAMPLE_RATE * 2):.2f} seconds"
                )
            await asyncio.sleep(2)

    def print_evt(self, evt, response):
        columns, rows = shutil.get_terminal_size()
        maxl = columns - 5
        print(str(evt)[:maxl] + " ...")
        if grounding := evt.get("serverContent", {}).get("groundingMetadata"):
            for chunk in grounding.get("groundingChunks", []):
                print(f"  <- {chunk.get("web").get("title")}")
        if parts := evt.get("serverContent", {}).get("modelTurn", {}).get("parts"):
            for part in parts:
                if part.get("inlineData") or part.get("text"):
                    continue
                print(f"  <- {part}")

    async def stdin_worker(self):
        def timeout_input(timeout: float = 0.1):
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.readline().rstrip()
            return None

        while self.running:
            try:
                line = await self.event_loop.run_in_executor(None, timeout_input)
                if line:
                    await self.send_text(line)
            except Exception as e:
                print(f"Exception: {e}")
                self.running = False

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

    async def screen_capture_worker(self):
        print(f"Screen capture enabled at {SCREEN_CAPTURE_FPS} frames per second.")
        with mss.mss(with_cursor=True) as sct:
            # Which display to capture. 0 is the composite of all screens. 1 is the primary screen.
            monitor = sct.monitors[1]
            while self.running:
                try:
                    frame = sct.grab(monitor)
                    # print(frame)
                    buffer = io.BytesIO()
                    Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX").save(
                        buffer, format="JPEG"
                    )
                    await self.send_video(buffer.getvalue())
                except Exception as e:
                    print(f"Exception: {e}")
                    self.running = False
                    return
                await asyncio.sleep(1.0 / SCREEN_CAPTURE_FPS)

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
