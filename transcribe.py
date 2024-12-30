import json
import os
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    chunk_size_ms = int(25 * 1024 * 1024 / (audio.frame_rate * audio.frame_width * audio.channels) * 1000)  # Convert bytes to milliseconds and ensure it's an integer
    chunks = [audio[i:i + chunk_size_ms] for i in range(0, len(audio), chunk_size_ms)]

    transcripts = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = f"chunk_{i}.mp3"
        chunk.export(chunk_file_path, format="mp3")
        with open(chunk_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            transcripts.append(transcription.text)
        os.remove(chunk_file_path)

    return transcripts

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python transcribe.py <path_to_audio_file> <output_file>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    result = transcribe_audio(audio_file_path)

    with open(output_file_path, "w") as output_file:
        json.dump(result, output_file, indent=4)

    print(f"Transcription saved to {output_file_path}")
