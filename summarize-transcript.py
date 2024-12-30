import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load transcript   
with open('transcript.json', 'r') as f:
    transcript = json.load(f)

transcript = transcript['utterances']

# Concatenate the utterances into a single string
transcript_text = "\n".join([f"Speaker {u['speaker']}: {u['text']}" for u in transcript])

# summarize transcript
response = client.chat.completions.create(model="gpt-4o",
messages=[
    {"role": "system", "content": "Summarize the following call transcript, highlighting key insights. Break the summary into sections: Personal Updates, Goals of the call, Key insights and realizations, important decisions made, and next steps. Any time a speaker learned something form the other, make sure that insight is tracked. This should be a detailed summary that includes direct quotes from speakers that are good representations of the summary. Try to use language similar to how the speakers speak."},
    {"role": "user", "content": transcript_text}
])

summary = response.choices[0].message.content
print(summary)

# save summary
with open('summary.txt', 'w') as f:
    f.write(summary)

