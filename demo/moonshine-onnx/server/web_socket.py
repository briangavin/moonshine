#Web Socket server on port 8765
import asyncio
import websockets
import threading

import argparse
import os
import time
from queue import Queue

import numpy as np
from silero_vad import VADIterator, load_silero_vad
from sounddevice import InputStream

from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

SAMPLING_RATE = 16000

CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MAX_LINE_LENGTH = 80

# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.1
VAD_THRESHOLD = 0.5



class Transcriber(object):
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text


def create_input_callback(q):
    """Callback method for sounddevice InputStream."""

    def input_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))

    return input_callback


def end_recording(speech, do_print=True):
    """Transcribes, prints and caches the caption then clears speech buffer."""
    text = transcribe(speech)
    #if do_print:
     #   print_captions(text)
    
    if len(text)>0:
        server.update_message(text)

    caption_cache.append(text)
    speech *= 0.0


def print_captions(text):
    """Prints right justified on same line, prepending cached captions."""
    if len(text) < MAX_LINE_LENGTH:
        for caption in caption_cache[::-1]:
            text = caption + " " + text
            if len(text) > MAX_LINE_LENGTH:
                break
    if len(text) > MAX_LINE_LENGTH:
        text = text[-MAX_LINE_LENGTH:]
    else:
        text = " " * (MAX_LINE_LENGTH - len(text)) + text
    print("\r" + (" " * MAX_LINE_LENGTH) + "\r" + text, end="", flush=True)


def soft_reset(vad_iterator):
    """Soft resets Silero VADIterator without affecting VAD model state."""
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.current_sample = 0


class WebSocketServer:
    def __init__(self):
        self.message_to_send = ""
        self.clients = set()

    async def register(self, websocket):
        self.clients.add(websocket)

    async def unregister(self, websocket):
        self.clients.remove(websocket)

    async def send_message(self, message):
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients])

    async def handle_connection(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                print(f"Received message: {message}")
        except websockets.ConnectionClosed:
            print("Connection closed")
        finally:
            await self.unregister(websocket)

    async def start(self):
        async with websockets.serve(self.handle_connection, "localhost", 8765):
            print("WebSocket server started on port 8765")
            await asyncio.Future()  # run forever

    def start_in_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self.start())
        threading.Thread(target=loop.run_forever).start()

    def update_message(self, message):
        self.message_to_send = message
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.send_message(message))


if __name__ == "__main__":
    server = WebSocketServer()
    server.start_in_thread()

    parser = argparse.ArgumentParser(
        prog="live_captions",
        description="Live captioning demo of Moonshine models",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=300,
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(q),
    )
    stream.start()

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    print("Press Ctrl+C to quit live captions.\n")

    with stream:
        print_captions("Ready...")
        try:
            while True:
                chunk, status = q.get()
                if status:
                    print(status)

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True
                        start_time = time.time()

                    if "end" in speech_dict and recording:
                        recording = False
                        end_recording(speech)

                elif recording:
                    # Possible speech truncation can cause hallucination.

                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech)
                        soft_reset(vad_iterator)

                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        message = transcribe(speech)
                        #server.update_message(message)
                        print_captions(message)
                        start_time = time.time()

        except KeyboardInterrupt:
            stream.close()

            if recording:
                while not q.empty():
                    chunk, _ = q.get()
                    speech = np.concatenate((speech, chunk))
                end_recording(speech, do_print=False)

            print(f"""

             model_name :  {model_name}
       MIN_REFRESH_SECS :  {MIN_REFRESH_SECS}s

      number inferences :  {transcribe.number_inferences}
    mean inference time :  {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
  model realtime factor :  {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
""")
            if caption_cache:
                print(f"Cached captions.\n{' '.join(caption_cache)}")

    # Example usage:
    #while True:
    #    message = input("Enter message to send to clients: ")
    #    server.update_message(message)