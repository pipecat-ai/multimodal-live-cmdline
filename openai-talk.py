import asyncio
import base64
import json
import os
import pyaudio
import queue
import shutil
import websockets


class AudioStreamer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()

    def mic_audio_in_callback(self, in_data, frame_count, time_info, status):
        payload = base64.b64encode(in_data).decode("utf-8")
        self.audio_queue.put(payload)
        return (None, pyaudio.paContinue)

    async def ws_send_worker(self):
        while True:
            while not self.audio_queue.empty():
                await self.ws.send(
                    json.dumps(
                        {"type": "input_audio_buffer.append", "audio": self.audio_queue.get()}
                    )
                )
            await asyncio.sleep(0.01)

    async def ws_receive_worker(self):
        async for m in self.ws:
            columns, rows = shutil.get_terminal_size()
            maxl = columns - 5
            print(m if len(m) <= maxl else (m[:maxl] + " ..."))
            evt = json.loads(m)
            if evt["type"] == "session.created":
                print("Connected: say something to GPT-4o")
                self.mic_audio_in.start_stream()
            elif evt["type"] == "response.audio.delta":
                audio = base64.b64decode(evt["delta"])
                self.speaker_audio_out.write(audio)

    async def run(self):
        self.mic_audio_in = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            input=True,
            stream_callback=self.mic_audio_in_callback,
            frames_per_buffer=int(24000 / 100) * 2,  # 20ms of audio
            start=False,
        )

        self.speaker_audio_out = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
        )

        self.ws = await websockets.connect(
            uri="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            additional_headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

        asyncio.create_task(self.ws_receive_worker())
        asyncio.create_task(self.ws_send_worker())

        await asyncio.sleep(15 * 60)


if __name__ == "__main__":
    asyncio.run(AudioStreamer().run())
