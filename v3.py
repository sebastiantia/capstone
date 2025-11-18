"""
Speech-to-Speech Companion Agent    
---------------------------------------

This module implements a real-time, context-aware conversational agent for elderly care
built on top of the Google Gemini 2.0 Flash Live API. It integrates low-latency
speech streaming, voice activity detection (VAD), transcription, and persistent memory
to enable natural, ongoing dialogue between a user and the AI.

Main Components
---------------
1. MemoryManager
   - Manages persistent `memory.json` storing past user and assistant utterances.
   - Performs asynchronous Whisper transcription for both sides.
   - Maintains a rolling conversation summary injected into the system prompt
     each session for continuity.

2. WavWriter
   - Buffers raw microphone PCM data using `webrtcvad` to detect voiced segments.
   - Writes per-utterance `.wav` files and enqueues them for transcription.
   - Prevents fragmentation via pre-buffering and silence-based rollover.

3. AudioLoop
   - Core orchestrator that manages the audio send/receive pipeline:
        • Captures microphone input and streams it to Gemini.
        • Receives streamed audio responses and plays them in real time.
        • Optionally suppresses echo by pausing mic capture during playback.
   - Handles async task orchestration using Python’s `asyncio.TaskGroup`.

System Behavior
---------------
- “Ferb” is the agent persona, designed to check in on elderly adults in a
  warm, conversational manner (see SYSTEM_PROMPT).
- The memory summary is appended to the model’s system prompt for each session,
  allowing continuity and adaptive responses over time.
- The pipeline uses 16 kHz for outgoing and 24 kHz for incoming audio.

Run Instructions
----------------
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install google-genai opencv-python pyaudio pillow mss openai-whisper webrtcvad
python v3.py
export GOOGLE_API_KEY=
```


TODO: 
Test performance on Raspberry compute
Accept startup user parameters: (name, age, hobbies, health conditions, robot personalization preference)
Echo suppression
Iterate on memory summarization strategy
User profile memory

Emulate caregiver's conversational style?
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
import whisper  
import webrtcvad
import wave, os, array
from collections import deque
from datetime import datetime

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    
    
SYSTEM_PROMPT = """
You are Ferb, an AI companion designed to support and check in on elderly adults living independently.
Your goals are:
1. Maintain friendly, respectful, and clear speech — never rushed or overly casual.
2. Keep conversations short and warm, asking gentle follow-up questions about how the person feels or what they’ve done today.
3. Track important well-being details (e.g., sleep, appetite, mood, pain, mobility, social activity).
4. Notice patterns — if someone repeatedly mentions fatigue, confusion, or loneliness, you should kindly acknowledge it.
5. Encourage healthy habits and social connection.
6. Avoid sounding like a medical professional; instead, act like a trusted friend who cares about their daily comfort.
7. Use simple language and a calm tone. Avoid jargon or long explanations.
8. When you recall past information, do it naturally (e.g., “Last week you said you slept better—how was last night?”).

You have access to a memory system that stores past conversations and events.
Use that memory to remember the patient’s preferences, routines, and emotional state.
"""

FORMAT = pyaudio.paInt16
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000

SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = int(SEND_SAMPLE_RATE * 0.02)  # 20 ms frame = 320 samples

MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "none"

client = genai.Client(http_options={"api_version": "v1beta"})

CONFIG = {"response_modalities": ["AUDIO"]}

pya = pyaudio.PyAudio()

MEMORY_FILE = "memory.json"

class MemoryManager:
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = Path(memory_file)
        self.user_queue = asyncio.Queue()
        self.assistant_queue = asyncio.Queue()
        self.model = whisper.load_model("small")
        self.summary = ""
        self.conversation_log = []
        self.load_memory()

    def load_memory(self):
        if self.memory_file.exists():
            try:
                content = self.memory_file.read_text().strip()
                if not content:  # Empty file
                    print("[Memory] Memory file is empty. Starting fresh.")
                    return
                data = json.loads(content)
                self.summary = data.get("summary", "")
                self.conversation_log = data.get("conversation_log", [])
                print(f"[Memory] Loaded {len(self.conversation_log)} turns from previous sessions.")
            except json.JSONDecodeError as e:
                print(f"[Memory] ⚠️ Invalid JSON in memory file: {e}. Starting fresh.")
        else:
            print("[Memory] No previous memory found.")

    def save_memory(self):
        self.memory_file.write_text(
            json.dumps(
                {"summary": self.summary, "conversation_log": self.conversation_log},
                indent=2,
            )
        )

    async def add_entry(self, speaker: str, text: str):
        """Store a conversation turn and update summary.""" 
        if not text.strip():
            return
        entry = {
            "speaker": speaker,
            "text": text.strip(),
            "timestamp": datetime.now().isoformat(),
        }
        self.conversation_log.append(entry)
        self.summary = (self.summary + f" {speaker}: {text}").strip()[-2000:]
        self.save_memory()

    async def transcribe_loop(self):
        """Runs two loops concurrently for user + assistant."""
        async def handle_queue(queue, speaker):
            while True:
                wav_path = await queue.get()
                try:
                    result = await asyncio.to_thread(self.model.transcribe, wav_path)
                    text = result["text"].strip()
                    if text:
                        print(f"[Memory] Transcribed {speaker}: {text[:80]}...")
                        await self.add_entry(speaker, text)
                except Exception as e:
                    print(f"[Memory] ⚠️ Transcription error for {speaker}: {e}")
                queue.task_done()

        await asyncio.gather(
            handle_queue(self.user_queue, "user"),
            handle_queue(self.assistant_queue, "assistant"),
        )

    def enqueue_audio(self, path: str, speaker="user"):
        """Add an audio file for transcription under correct speaker."""
        if speaker == "assistant":
            asyncio.create_task(self.assistant_queue.put(path))
        else:
            asyncio.create_task(self.user_queue.put(path))

class WavWriter:
    """Handles streamed PCM audio and writes separate WAVs for each spoken utterance using VAD."""

    def __init__(
        self,
        pya,
        fmt,
        channels,
        rate,
        chunk_size,
        out_dir="recordings",
        vad_aggressiveness=3,
        silence_limit_sec=1.5,
        pre_buffer_sec=0.5,
    ):
        self.pya = pya
        self.format = fmt
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # --- VAD setup ---
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_ms = int(chunk_size / rate * 1000)
        if self.frame_ms not in (10, 20, 30):
            print(f"[WavWriter] ⚠️ CHUNK_SIZE should represent 10/20/30 ms, current={self.frame_ms} ms")

        # --- internal state ---
        self._writer = None
        self._filename = None
        self._silence_chunks = 0
        self._pre_buffer = deque(maxlen=int(rate / chunk_size * pre_buffer_sec))
        self._silence_limit_chunks = int(rate / chunk_size * silence_limit_sec)
        self.memory_manager = None  # optional hook

    # ---------- helpers ----------
    def _new_file(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filename = os.path.join(self.out_dir, f"user_{ts}.wav")
        wf = wave.open(self._filename, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pya.get_sample_size(self.format))
        wf.setframerate(self.rate)
        self._writer = wf
        print(f"[WavWriter] 🎙️ Start → {self._filename}")
        for buf in self._pre_buffer:
            self._writer.writeframes(buf)
        self._pre_buffer.clear()

    def _rollover(self):
        if self._writer:
            self._writer.close()
            print(f"[WavWriter] 💾 Saved {self._filename}")
            if self.memory_manager:
                self.memory_manager.enqueue_audio(self._filename)
        self._writer = None
        self._filename = None
        self._silence_chunks = 0

    # ---------- main ----------
    def write(self, data: bytes):
        self._pre_buffer.append(data)
        is_speech = self.vad.is_speech(data, self.rate)

        if is_speech:
            if not self._writer:
                self._new_file()
            self._writer.writeframes(data)
            self._silence_chunks = 0
        else:
            if self._writer:
                self._writer.writeframes(data)
                self._silence_chunks += 1
                if self._silence_chunks > self._silence_limit_chunks:
                    self._rollover()

    def close(self):
        if self._writer:
            self._writer.close()
            print(f"[WavWriter] Closed {self._filename}")
            
            
class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, suppress_echo=False):
        self.video_mode = video_mode
        self.suppress_echo = suppress_echo

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.memory_manager = MemoryManager()
        
        self._assistant_done_talking = asyncio.Event()

        

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
        
        writer = WavWriter(pya, FORMAT, CHANNELS, SEND_SAMPLE_RATE, CHUNK_SIZE, vad_aggressiveness=2)
        writer.memory_manager = self.memory_manager

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        try:
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)

                # 🧩 Soft echo suppression
                if self.suppress_echo and not self._assistant_done_talking.is_set():
                    continue  # skip sending frames while assistant is talking

                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                writer.write(data)
        finally:
            writer.close()

    async def receive_audio(self):
        """
        Receives Gemini's streamed audio and merges each full reply into one WAV
        before transcription.
        """
        while True:
            turn = self.session.receive()
            audio_chunks = []
            async for response in turn:
                if data := response.data:
                    if self.suppress_echo:
                        self._assistant_done_talking.clear()  # 🚫 pause listening
                    # Stream to speaker immediately
                    self.audio_in_queue.put_nowait(data)
                    audio_chunks.append(data)
                    continue

            # ---- Model finished speaking (end of turn) ----
            if audio_chunks:
                if self.suppress_echo:
                    await asyncio.sleep(0.25)  # small grace delay to avoid residual echo
                    self._assistant_done_talking.set()  # ✅ resume listening
                
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = f"assistant_{ts}.wav"
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pya.get_sample_size(FORMAT))
                    wf.setframerate(RECEIVE_SAMPLE_RATE)
                    for chunk in audio_chunks:
                        wf.writeframes(chunk)

                print(f"[Assistant Audio] 💾 Saved combined reply to {wav_path}")
                self.memory_manager.enqueue_audio(wav_path, speaker="assistant")


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
                   config = {
                        **CONFIG,
                        "system_instruction": f"{SYSTEM_PROMPT}\n\n<>Known memory summary:<>\n{self.memory_manager.summary}",
                    }
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                
                # ✅ Allow listening initially
                self._assistant_done_talking.set()

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
    
    parser.add_argument(
        "--suppress-echo",
        action="store_true",
        help="Temporarily pause microphone capture while the assistant is speaking."
    )

    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())