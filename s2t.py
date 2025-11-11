#!/usr/bin/env python3
"""
mic_vosk_stream.py
Realtime microphone -> Vosk streaming transcription.
Usage:
  python mic_vosk_stream.py --model ./model/<your-model-dir>
"""

import argparse
import queue
import sys
import json
import os
import sounddevice as sd
from vosk import Model, KaldiRecognizer

def main():
    p = argparse.ArgumentParser(description="Microphone -> Vosk realtime")
    p.add_argument("--model", "-m", required=True, help="Path to vosk model directory")
    p.add_argument("--device", "-d", type=int, default=None, help="Optional input device id")
    p.add_argument("--samplerate", "-r", type=int, default=None, help="Optional samplerate (Hz)")
    args = p.parse_args()

    if not os.path.isdir(args.model):
        print(f"model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    model = Model(args.model)

    # choose device and samplerate
    device = args.device
    if args.samplerate:
        samplerate = args.samplerate
    else:
        try:
            info = sd.query_devices(device, 'input')
            samplerate = int(info['default_samplerate'])
        except Exception:
            samplerate = 16000

    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                               dtype='int16', channels=1, callback=callback, device=device):
            print("listening (ctrl-c to stop)...", file=sys.stderr)
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "")
                    if text:
                        # final result
                        print(text)
                else:
                    # partial result (updated in-place)
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    if partial:
                        sys.stdout.write("\r" + partial)
                        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    except Exception as e:
        print("error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
