def get_current_weather(
    location: str,
):
    """Returns the current weather.

    Args:
      location: The location to get the weather for.
    """
    return {
        "status": "success",
        "response": f"The current weather in {location} is 72 degrees with scattered thunderstorms.",
    }


def line_printer(line: str):
    """Prints a line to the console.

    Args:
      line: The line to print.
    """
    print(f"  \033[1m :: {line} ::\033[0m")
    return {"status": "success"}
