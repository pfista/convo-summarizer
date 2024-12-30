from dotenv import load_dotenv
load_dotenv()
import assemblyai as aai
import argparse
import json
import os

# Replace with your API key
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Set up argument parser
parser = argparse.ArgumentParser(description='Transcribe an audio file using AssemblyAI.')
parser.add_argument('file_path', type=str, help='Path to the audio file to transcribe')
args = parser.parse_args()

# Use the file path from the command-line argument
FILE_PATH = args.file_path

config = aai.TranscriptionConfig(speaker_labels=True)
transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_PATH, config=config)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")

# Convert the transcript to a dictionary
transcript_dict = {
    'status': transcript.status,
    'utterances': [
        {
            'speaker': utterance.speaker,
            'text': utterance.text
        } for utterance in transcript.utterances
    ]
}

# Save the transcript to a file
with open('transcript.json', 'w') as f:
    json.dump(transcript_dict, f, indent=2)
    print(f"Transcript saved to transcript.json")