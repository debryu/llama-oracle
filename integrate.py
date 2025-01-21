import ollama

response = ollama.chat(
  model = "x/llama3.2-vision",
  messages = [
    {"role": "user", "content": "What is depicted inside the image?", "images": ["./images/cat.png"]},
  ]
)

print(response["message"]['role'])
print(response["message"]['content'])
