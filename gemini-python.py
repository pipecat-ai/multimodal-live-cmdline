# pip install google-genai

import argparse
import asyncio
import os
import pyaudio
import queue
import re
import shutil
import websockets

from google import genai

MODEL = "models/gemini-2.0-flash-exp"
HOST = "generativelanguage.googleapis.com"
API_KEY = os.getenv("GEMINI_API_KEY")

MIC_SAMPLE_RATE = 16000
SPEAKER_SAMPLE_RATE = 24000
FORMAT = "S16_LE"
CHANNELS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini Talk with optional search functionality")
    args = parser.parse_args()


class AudioStreamer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.running = False
        self.client = None
        self.session = None

        self.mic_audio_in_stream = None
        self.mic_audio_queue = queue.Queue()  # base-64 encoded audio chunks

        self.speaker_audio_buffer = bytearray()  # raw audio bytes

    async def mic_audio_sender(self):
        event_loop = asyncio.get_event_loop()

        def mic_audio_in_callback(in_data, frame_count, time_info, status):
            nonlocal self
            nonlocal event_loop
            if not self.running:
                return (None, pyaudio.paComplete)
            if self.session:
                payload = in_data
                # print(f"> sending {len(payload)} audio bytes")
                event_loop.create_task(
                    self.session.send({"data": payload, "mime_type": "audio/pcm"})
                )
            return (None, pyaudio.paContinue)

        self.mic_audio_in_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=MIC_SAMPLE_RATE,
            input=True,
            stream_callback=mic_audio_in_callback,
            frames_per_buffer=int(MIC_SAMPLE_RATE / 1000) * 2 * 50,  # 50ms (S16_LE is 2 bytes)
        )
        print("started mic audio in")

    async def speaker_audio_writer(self):
        def speaker_audio_out_callback(in_data, frame_count, time_info, status):
            if not self.running:
                return (bytes(frame_count * CHANNELS * 2), pyaudio.paComplete)
            audio = bytes(self.speaker_audio_buffer[: frame_count * CHANNELS * 2])
            del self.speaker_audio_buffer[: frame_count * CHANNELS * 2]
            audio += b"\0" * (frame_count * CHANNELS * 2 - len(audio))
            return (audio, pyaudio.paContinue)

        self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SPEAKER_SAMPLE_RATE,
            output=True,
            frames_per_buffer=256,
            stream_callback=speaker_audio_out_callback,
        )

    def print_evt(self, response):
        columns, rows = shutil.get_terminal_size()
        maxl = columns - 5
        print(str(response.model_dump(exclude_unset=True, exclude_none=True))[:maxl] + " ...")

    async def receive_from_gemini(self):
        while self.running:
            try:
                async for evt in self.session.receive():
                    self.print_evt(evt)
                    if evt.server_content:
                        if evt.server_content.turn_complete:
                            # SDK does not pass the interrupted event through, so we have to assume
                            # this might be an interruption and clear the speaker buffer.
                            self.speaker_audio_buffer.clear()
                            continue
                        if (
                            evt.server_content.model_turn
                            and evt.server_content.model_turn.parts[0].inline_data
                        ):
                            inline_data = evt.server_content.model_turn.parts[0].inline_data
                            mime_str = inline_data.mime_type
                            mime_type, sample_rate = re.match(
                                r"([\w/]+);rate=(\d+)", mime_str
                            ).groups()
                            if mime_type == "audio/pcm" and sample_rate == str(SPEAKER_SAMPLE_RATE):
                                self.speaker_audio_buffer.extend(inline_data.data)
                            else:
                                print(f"Unsupported mime type or sample rate: {mime_str}")
            except asyncio.CancelledError:
                pass
            except websockets.exceptions.ConnectionClosedOK:
                pass
            except Exception as e:
                print(f"Exception: {e}")
                self.running = False

    async def cleanup(self):
        self.running = False
        await self.session.close()
        self.p.terminate()

    async def run(self):
        self.running = True

        client = genai.Client(
            http_options={
                "api_version": "v1alpha",
                "url": "generativelanguage.googleapis.com",
            }
        )
        config = {
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": "Charon",
            },
            "tools": [{"google_search": {}}],
        }

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            self.session = session

            asyncio.create_task(self.mic_audio_sender())
            asyncio.create_task(self.speaker_audio_writer())
            asyncio.create_task(self.receive_from_gemini())

            # message = "Can you use google search tell me about the largest earthquake in california the week of Dec 5 2024?"
            # print("> ", message, "\n")

            # await session.send(message, end_of_turn=True)
            # await session.send("", end_of_turn=True)

            try:
                while self.running:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                await self.cleanup()
                pass


if __name__ == "__main__":
    # parse_args()
    asyncio.run(AudioStreamer().run())
