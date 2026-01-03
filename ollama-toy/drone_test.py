import json
from ollama import chat

# Dummy Tello Interface
class DummyDrone:
    def __init__(self):
        self.state = "grounded"

    def send_command(self, command: str):
        print(f"[DUMMY] Executing Command: {command}")
        return "ok"

    def takeoff(self):
        if self.state == "grounded":
            self.send_command("takeoff")
            self.state = "flying"
        else:
            print("[DUMMY] Already flying.")

    def land(self):
        if self.state == "flying":
            self.send_command("land")
            self.state = "grounded"
        else:
            print("[DUMMY] Already grounded.")

drone = DummyDrone()

# 1) Define your actual functions
def takeoff():
    return drone.takeoff()

def land():
    return drone.land()

def move(direction: str, distance: int):
    return drone.send_command(f"{direction} {distance}")

def rotate(direction: str, degrees: int):
    return drone.send_command(f"{direction} {degrees}")

# 2) Describe them as tools
# ... (tools definition remains same) ...
# ... (messages and chat call remain same) ...

# ... (inside the loop) ...


# 2) Describe them as tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "takeoff",
            "description": "Take off the drone.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "land",
            "description": "Land the drone.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the drone in a direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "'up', 'down', 'left', 'right', 'forward', 'back'",
                        "enum": ["up", "down", "left", "right", "forward", "back"]
                    },
                    "distance": {"type": "integer", "description": "Distance in cm"},
                },
                "required": ["direction", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": "Rotate the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "description": "'cw' or 'ccw'", "enum": ["cw", "ccw"]},
                    "degrees": {"type": "integer", "description": "Degrees (1-360)"},
                },
                "required": ["direction", "degrees"],
            },
        },
    },
]

# 3) Call FunctionGemma with a natural language request
messages = [
    {"role": "system", "content": "You are a helpful assistant. You must perform every single action requested by the user. If the user requests a sequence of actions (e.g. 'do X then do Y'), generate tool calls for ALL of them in order. Do not miss any steps."},
    # Few-shot example
    {"role": "user", "content": "Move up 50cm and then rotate cw 90 degrees."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "move", "arguments": {"direction": "up", "distance": 50}}},
        {"function": {"name": "rotate", "arguments": {"direction": "cw", "degrees": 90}}}
    ]},
    # Actual user prompt
    {"role": "user", "content":  "I want you to move forward 100cm and then land."}
]

print(f"User Prompt: {messages[-1]['content']}")

response = chat(
    # model="functiongemma:270m",
    # model="llama4:latest",
    model="nemotron-3-nano:latest",
    messages=messages,
    tools=tools,
    options={"temperature": 0.0},
)

print("Raw response:", response)

# 4) Check for tool calls
tool_calls = response["message"].get("tool_calls", [])

if tool_calls:
    print(f"Found {len(tool_calls)} tool calls:")
    for call in tool_calls:
        func_name = call["function"]["name"]
        args = call["function"]["arguments"]

        print(f"Calling: {func_name}({args})")

        # --- Centralized State Validation ---
        if func_name in ["move", "rotate"] and drone.state == "grounded":
            print(f"[Agent] Error: Cannot {func_name} while grounded. Correcting: Taking off...")
            takeoff()

        if func_name == "takeoff" and drone.state == "flying":
            print("[Agent] Warning: Already flying. Skipping takeoff.")
            continue # Skip execution

        if func_name == "land" and drone.state == "grounded":
            print("[Agent] Warning: Already grounded. Skipping land.")
            continue # Skip execution
        # ------------------------------------

        result = None
        # 5) Execute your real function
        if func_name == "takeoff":
            result = takeoff()
        elif func_name == "land":
            result = land()
        elif func_name == "move":
            result = move(**args)
        elif func_name == "rotate":
            result = rotate(**args)

        # 6) Append the tool result to history
        messages.append({
            "role": "tool",
            "content": str(result),
            "name": func_name,
        })
else:
    # No tool call, just a normal reply
    print("Assistant:", response["message"]["content"])
