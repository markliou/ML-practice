import json
from ollama import chat

# 1) Define your actual function
def get_weather(city: str) -> str:
    return f"It is sunny in {city}."

# 2) Describe it as a tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]

# 3) Call FunctionGemma with a natural language request
messages = [
    {"role": "user", "content": "What is the weather in Paris?"}
]

response = chat(
    model="functiongemma:270m",
    messages=messages,
    tools=tools,
)

print("Raw response:", response)

# 4) Check for tool calls
tool_calls = response["message"].get("tool_calls", [])
if tool_calls:
    call = tool_calls[0]
    func_name = call["function"]["name"]
    args = call["function"]["arguments"]

    print(f"Calling: {func_name}({args})")

    # 5) Execute your real function
    if func_name == "get_weather":
        result = get_weather(**args)

    print("Tool result:", result)
else:
    # No tool call, just a normal reply
    print("Assistant:", response["message"]["content"])
