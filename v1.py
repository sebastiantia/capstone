# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss openai-whisper
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import wave
import array
from datetime import datetime
import pyaudio
from collections import deque
import argparse
import json
import asyncio
import tempfile
import threading
from pathlib import Path
import whisper  # pip install openai-whisper

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "none"

client = genai.Client(http_options={"api_version": "v1beta"})

CONFIG = {"response_modalities": ["AUDIO"]}

pya = pyaudio.PyAudio()


MEMORY_FILE = "memory.json"

class MemoryManager:
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = Path(memory_file)
        self.transcribe_queue = asyncio.Queue()
        self.model = whisper.load_model("tiny")  # Fastest for MVP
        self.summary = ""
        self.load_memory()

    def load_memory(self):
        if self.memory_file.exists():
            data = json.loads(self.memory_file.read_text())
            self.summary = data.get("summary", "")
            print(f"[Memory] Loaded previous memory ({len(self.summary)} chars)")
        else:
            print("[Memory] No previous memory found")

    def save_memory(self):
        self.memory_file.write_text(json.dumps({"summary": self.summary}, indent=2))

    async def transcribe_loop(self):
        while True:
            wav_path = await self.transcribe_queue.get()
            try:
                result = await asyncio.to_thread(self.model.transcribe, wav_path)
                text = result["text"].strip()
                if text:
                    print(f"[Memory] Transcribed: {text[:80]}...")
                    await self.update_summary(text)
            except Exception as e:
                print(f"[Memory] ⚠️ Transcription error for {wav_path}: {e}")
            self.transcribe_queue.task_done()

    async def update_summary(self, new_text):
        # Simple rolling summary logic — can replace with LLM later
        self.summary = (self.summary + " " + new_text).strip()[-2000:]
        self.save_memory()

    def enqueue_audio(self, path: str):
        asyncio.create_task(self.transcribe_queue.put(path))


class WavWriter:
    """Handles writing streamed PCM audio to sequential WAV files,
    with adaptive noise floor and hysteresis-based silence detection."""

    def __init__(
        self,
        pya,
        fmt,
        channels,
        rate,
        chunk_size,
        out_dir="recordings",
        silence_limit_sec=2.0,
        min_speech_sec=0.5,
        calibrate_sec=1.0,
    ):
        self.pya = pya
        self.format = fmt
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.silence_limit_chunks = int(rate / chunk_size * silence_limit_sec)
        self.min_speech_chunks = int(rate / chunk_size * min_speech_sec)
        self.calibrate_chunks = int(rate / chunk_size * calibrate_sec)

        self._writer = None
        self._filename = None
        self._silence_chunks = 0
        self._speech_chunks = 0
        self._noise_floor = 500  # temporary default
        self._pre_buffer = deque(maxlen=10)  # ~0.6 s of audio

        self._calibrated = False
        self._calibration_data = []

    # ---------- internal helpers ----------
    def _avg_amp(self, data):
        samples = array.array("h", data)
        return sum(abs(s) for s in samples) / len(samples)

    def _new_file(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filename = os.path.join(self.out_dir, f"user_{ts}.wav")
        wf = wave.open(self._filename, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pya.get_sample_size(self.format))
        wf.setframerate(self.rate)
        self._writer = wf
        print(f"[WavWriter] 🎙️ Recording → {self._filename}")

        # write pre-buffered frames to avoid clipping start
        for buf in self._pre_buffer:
            self._writer.writeframes(buf)
        self._pre_buffer.clear()

    def _rollover(self):
        if self._writer:
            self._writer.close()
            print(f"[WavWriter] 💾 Saved {self._filename}")
            if hasattr(self, "memory_manager"):
                self.memory_manager.enqueue_audio(self._filename)
        self._writer = None
        self._filename = None
        self._silence_chunks = 0
        self._speech_chunks = 0

    # ---------- public ----------
    def write(self, data: bytes):
        # 1. Calibration phase (first ~1 s)
        if not self._calibrated:
            self._calibration_data.append(self._avg_amp(data))
            if len(self._calibration_data) >= self.calibrate_chunks:
                self._noise_floor = max(300, sum(self._calibration_data) / len(self._calibration_data))
                self._calibrated = True
                print(f"[WavWriter] Noise floor calibrated: {self._noise_floor:.0f}")
            return

        # 2. Compute amplitude & thresholds
        avg_amp = self._avg_amp(data)
        start_thresh = self._noise_floor * 3.0
        stop_thresh = self._noise_floor * 1.2  # <- slightly lower than before (was 1.5)

        # 3. Always keep a small pre-buffer
        self._pre_buffer.append(data)

        # 4. Speech start / continue
        if avg_amp > start_thresh:
            if not self._writer:
                self._new_file()
            self._writer.writeframes(data)
            self._silence_chunks = 0
            self._speech_chunks += 1
            return

        # 5. Speech ongoing but dropping to low amplitude
        if self._writer:
            self._writer.writeframes(data)
            
            # increment silence counter only if we're *consistently* quiet
            if avg_amp < stop_thresh:
                self._silence_chunks += 1
            else:
                self._silence_chunks = 0

            # wait longer before deciding to close
            if (
                self._silence_chunks > self.silence_limit_chunks * 1.5  # <- longer wait
                and self._speech_chunks >= self.min_speech_chunks
            ):
                self._rollover()
    def close(self):
        if self._writer:
            self._writer.close()
            print(f"[WavWriter] Closed {self._filename}")

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.memory_manager = MemoryManager()
        

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    # def _get_frame(self, cap):
    #     # Read the frameq
    #     ret, frame = cap.read()
    #     # Check if the frame was read successfully
    #     if not ret:
    #         return None
    #     # Fix: Convert BGR to RGB color space
    #     # OpenCV captures in BGR but PIL expects RGB format
    #     # This prevents the blue tint in the video feed
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
    #     img.thumbnail([1024, 1024])

    #     image_io = io.BytesIO()
    #     img.save(image_io, format="jpeg")
    #     image_io.seek(0)

    #     mime_type = "image/jpeg"
    #     image_bytes = image_io.read()
    #     return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    # async def get_frames(self):
    #     # This takes about a second, and will block the whole program
    #     # causing the audio pipeline to overflow if you don't to_thread it.
    #     cap = await asyncio.to_thread(
    #         cv2.VideoCapture, 0
    #     )  # 0 represents the default camera

    #     while True:
    #         frame = await asyncio.to_thread(self._get_frame, cap)
    #         if frame is None:
    #             break

    #         await asyncio.sleep(1.0)

    #         await self.out_queue.put(frame)

    #     # Release the VideoCapture object
    #     cap.release()

    # def _get_screen(self):
    #     sct = mss.mss()
    #     monitor = sct.monitors[0]

    #     i = sct.grab(monitor)

    #     mime_type = "image/jpeg"
    #     image_bytes = mss.tools.to_png(i.rgb, i.size)
    #     img = PIL.Image.open(io.BytesIO(image_bytes))

    #     image_io = io.BytesIO()
    #     img.save(image_io, format="jpeg")
    #     image_io.seek(0)

    #     image_bytes = image_io.read()
    #     return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    # async def get_screen(self):

    #     while True:
    #         frame = await asyncio.to_thread(self._get_screen)
    #         if frame is None:
    #             break

    #         await asyncio.sleep(1.0)

    #         await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        writer = WavWriter(pya, FORMAT, CHANNELS, SEND_SAMPLE_RATE, CHUNK_SIZE)
        writer.memory_manager = self.memory_manager

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        try:
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                writer.write(data)
        finally:
            writer.close()

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(
                    model=MODEL,
                    config={**CONFIG, "system_instruction": self.memory_manager.summary or "You are a helpful companion robot."}
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                listening_task = tg.create_task(self.listen_audio())
                # if self.video_mode == "camera":
                #     tg.create_task(self.get_frames())
                # elif self.video_mode == "screen":
                #     tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.memory_manager.transcribe_loop())

                
                print("seb debug: tasks started successfully")
                await listening_task
                
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"], # LIVE API 
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())