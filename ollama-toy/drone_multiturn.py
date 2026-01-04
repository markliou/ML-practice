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
        return "ok"

    def land(self):
        if self.state == "flying":
            self.send_command("land")
            self.state = "grounded"
        else:
            print("[DUMMY] Already grounded.")
        return "ok"

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

# 3) Initialize Conversation History
messages = [
    {"role": "system", "content": "You are a helpful assistant controlling a drone. You expect clear commands. If a user asks for multiple actions, call them in order. Always double check if you can perform the action based on the drone's likely state (e.g. must take off before moving)."},
]

print("--- Multi-turn Drone Control ---")
print("Type 'exit' or 'quit' to stop.")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    while True:
        response = chat(
            # model="llama4:latest",
            model="functiongemma",
            messages=messages,
            tools=tools,
        )

        message_content = response["message"].get("content")
        tool_calls = response["message"].get("tool_calls", [])

        # Always add the assistant's response to history, whether it's content or tool calls
        messages.append(response["message"])

        if tool_calls:
            print(f"[Assistant] processing {len(tool_calls)} tool calls...")
            for call in tool_calls:
                func_name = call["function"]["name"]
                args = call["function"]["arguments"]

                print(f"  > Calling: {func_name}({args})")

                # State validation (optional but good for robustness)
                if func_name in ["move", "rotate"] and drone.state == "grounded":
                    print(f"    [System] Error: Cannot {func_name} while grounded. Auto-Taking off...")
                    takeoff()

                result = None
                if func_name == "takeoff":
                    result = takeoff()
                elif func_name == "land":
                    result = land()
                elif func_name == "move":
                    if "direction" in args and "distance" in args:
                        result = move(**args)
                    else:
                        result = "Error: Missing arguments for move"
                elif func_name == "rotate":
                    if "direction" in args and "degrees" in args:
                        result = rotate(**args)
                    else:
                        result = "Error: Missing arguments for rotate"
                else:
                    result = f"Error: Unknown tool {func_name}"

                # Append tool result to history so the model knows what happened
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "name": func_name,
                })

            # After handling tool calls, we loop back to let the model comment on the results or do more things
            # The model will see the "tool" messages we just added.
            continue
        else:
            # No tool calls, just text response.
            if message_content:
                print(f"Assistant: {message_content}")

            # Break inner loop to prompt user again
            break
