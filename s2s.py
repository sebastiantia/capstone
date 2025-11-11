import asyncio
import sys
import os
import wave
import logging
import sounddevice as sd
from google import genai
from google.genai import types

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")

# Make sure to set your GOOGLE_API_KEY environment variable
# export GOOGLE_API_KEY="your_api_key"
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful assistant and answer in a friendly tone.",
}

# microphone parameters — now match the model (24kHz, 16-bit PCM mono)
SAMPLERATE = 24000
BLOCKSIZE  = 8000
CHANNELS   = 1


async def mic_to_gemini():
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue[bytes] = asyncio.Queue()

    # input callback (sounddevice runs in another thread)
    def input_callback(indata, frames, time_info, status):
        if status:
            logging.warning("input status: %s", status)
        loop.call_soon_threadsafe(aq.put_nowait, bytes(indata))

    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        # prepare output WAV file and speaker stream
        wf = wave.open("audio_out.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLERATE)

        speaker_stream = sd.RawOutputStream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype="int16",
            channels=1
        )

        async def send_mic():
            with sd.RawInputStream(
                samplerate=SAMPLERATE,
                blocksize=BLOCKSIZE,
                dtype="int16",
                channels=CHANNELS,
                callback=input_callback
            ):
                logging.info("listening... (ctrl-c to stop)")
                while True:
                    data = await aq.get()
                    if not data:
                        continue
                    logging.debug("mic: got %d bytes", len(data))
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=data,
                            mime_type=f"audio/pcm;rate={SAMPLERATE}"
                        )
                    )

        async def recv_model():
            async for response in session.receive():
                logging.debug("received response: %r", response)
                data = getattr(response, "data", None)
                if data:
                    wf.writeframes(data)
                    await loop.run_in_executor(None, speaker_stream.write, data)

        try:
            speaker_stream.start()
            sender = asyncio.create_task(send_mic(), name="sender")
            receiver = asyncio.create_task(recv_model(), name="receiver")

            done, pending = await asyncio.wait(
                {sender, receiver},
                return_when=asyncio.FIRST_EXCEPTION
            )
            for t in done:
                exc = t.exception()
                if exc:
                    raise exc

        finally:
            speaker_stream.stop()
            speaker_stream.close()
            wf.close()


if __name__ == "__main__":
    try:
        asyncio.run(mic_to_gemini())
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    except Exception:
        logging.exception("fatal error")


# BENCHMARK RESULT 