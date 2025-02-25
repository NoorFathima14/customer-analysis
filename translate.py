import requests
import json

url = "http://localhost:11434/api/generate"
payload = {
    "model": "lauchacarro/qwen2.5-translator",
    "prompt": "Translate the following English text to French: 'Hello, World!'",
    "stream": True  
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

final_output = ""
for line in response.iter_lines(decode_unicode=True):
    if line:  
        try:
            token_data = json.loads(line)
            final_output += token_data.get("response", "")
            if token_data.get("done", False): # works because the last token is sent with {"done": "true"} to indicate end of response
                break
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            continue

print(final_output)
