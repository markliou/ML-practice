import ollama
import json

# Dummy Tello Interface for testing interpretation
class DummyDrone:
    def send_command(self, command: str):
        print(f"[DUMMY] Executing Command: {command}")
        return "ok"

drone = DummyDrone()

# Tool Definitions
def takeoff():
    """Take off the drone."""
    return drone.send_command("takeoff")

def land():
    """Land the drone."""
    return drone.send_command("land")

def move(direction: str, distance: int):
    """
    Move the drone in a direction.
    :param direction: 'up', 'down', 'left', 'right', 'forward', 'back'
    :param distance: Distance in cm
    """
    return drone.send_command(f"{direction} {distance}")

def rotate(direction: str, degrees: int):
    """
    Rotate the drone.
    :param direction: 'cw' or 'ccw'
    :param degrees: Degrees (1-3600)
    """
    return drone.send_command(f"{direction} {degrees}")

names_to_functions = {
    'takeoff': takeoff,
    'land': land,
    'move': move,
    'rotate': rotate,
}

def test_interpretation(prompt: str):
    print(f"\n--- Testing Interpretation ---")
    print(f"User Prompt: \"{prompt}\" ")

    # Using functiongemma:270m
    model = 'functiongemma:270m'

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            tools=[takeoff, land, move, rotate],
        )

        tool_calls = response.get('message', {}).get('tool_calls', [])

        if tool_calls:
            print(f"Found {len(tool_calls)} tool calls:")
            for tool in tool_calls:
                name = tool['function']['name']
                args = tool['function']['arguments']
                print(f"  -> Calling: {name}({args})")

                if name in names_to_functions:
                    names_to_functions[name](**args)
        else:
            print("No tool calls detected.")
            print("Response text:", response['message']['content'])

    except Exception as e:
        print(f"Error during Ollama call: {e}")

if __name__ == "__main__":
    # Test cases
    test_interpretation("Please take off and then land immediately.")
    test_interpretation("Fly forward 100cm, turn 90 degrees clockwise, then fly back 50cm.")
    test_interpretation("Go up 200, then land.")
