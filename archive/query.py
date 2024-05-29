import requests

url = "https://general-runtime.voiceflow.com/knowledge-base/query"

payload = {
    "question": "Who is Ian?",
    "chunkLimit": 2,
    "synthesis": True,
    "settings": {
        "model": "claude-instant-v1",
        "temperature": 0.1,
        "system": "You are an AI FAQ assistant. Information will be provided to help answer the user's questions. Always summarize your response to be as brief as possible and be extremely concise. Your responses should be fewer than a couple of sentences. Do not reference the material provided in your response."
    }
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "VF.DM.65c5cd65a0c117f719c36177.NXM1ocQfxv39x7kX"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)