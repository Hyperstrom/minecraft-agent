"""
tool_registry.py — All tools the LLM can call, with descriptions and param schemas.
"""

TOOLS = [
    {
        "name":        "MOVE",
        "description": "Move in a cardinal direction",
        "params":      {"direction": "north|south|east|west", "distance": "int (blocks)"},
    },
    {
        "name":        "MINE",
        "description": "Dig the nearest block of given type",
        "params":      {"block": "str (e.g. oak_log, iron_ore, stone)"},
    },
    {
        "name":        "COLLECT",
        "description": "Pick up dropped items from the ground",
        "params":      {"item": "str", "quantity": "int"},
    },
    {
        "name":        "CRAFT",
        "description": "Craft an item using inventory materials",
        "params":      {"item": "str", "count": "int"},
    },
    {
        "name":        "EAT",
        "description": "Eat a food item from inventory",
        "params":      {"item": "str (food name)"},
    },
    {
        "name":        "CHAT",
        "description": "Send a chat message in-game",
        "params":      {"message": "str"},
    },
    {
        "name":        "FOLLOW",
        "description": "Follow a nearby player continuously",
        "params":      {"player": "str (username)"},
    },
    {
        "name":        "GOTO",
        "description": "Walk to exact coordinates",
        "params":      {"x": "int", "y": "int", "z": "int"},
    },
    {
        "name":        "STOP",
        "description": "Stop all movement immediately",
        "params":      {},
    },
    {
        "name":        "IDLE",
        "description": "Do nothing, observe the world",
        "params":      {},
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS}


def get_tool_schema_text() -> str:
    """Return a compact text listing of all tools for the system prompt."""
    lines = []
    for t in TOOLS:
        if t["params"]:
            params = ", ".join(f'{k}: {v}' for k, v in t["params"].items())
        else:
            params = "no params"
        lines.append(f'  {t["name"]}: {t["description"]} ({params})')
    return "\n".join(lines)
