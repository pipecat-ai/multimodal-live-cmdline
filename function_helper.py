import importlib.util
import inspect
import types

from typing import Any, Callable, Coroutine, List, Tuple, Union

from google import genai

from_function = genai.types.FunctionDeclaration.from_function


def create_function_declarations_from_file(
    filename: str,
) -> Tuple[[genai.types.FunctionDeclaration], types.ModuleType]:
    """
    Import a Python file and return function declarations automatically generated
    by the google-genai SDK. Also return the module object.

    Args:
        filename (str): Path to the Python file to import

    Returns:
        tuple: (list of function objects, module object)
    """

    functions, module = _import_functions_from_file(filename)

    # We need a google-genai client object to use the from_function utility method. (Mostly
    # because internally the it does slightly different things for Google AI Studio and Google
    # Vertex AI connections.)
    client = genai.Client(
        http_options={
            "api_version": "v1alpha",
            "url": "generativelanguage.googleapis.com",
        }
    )

    function_declarations = []
    for func in functions:
        function_declarations.append(
            from_function(client, func).model_dump(exclude_unset=True, exclude_none=True)
        )

    return function_declarations, module


async def call_function(module: Any, function_name: str, **kwargs) -> Any:
    """
    Call a function from a module by its name, handling both sync and async
    functions.

    Args:
        module: The module object containing the function
        function_name: The name of the function to call
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        AttributeError: If the function doesn't exist in the module
    """
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in module")

    func = getattr(module, function_name)
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    return func(**kwargs)


def _import_functions_from_file(
    filename: str,
) -> Tuple[List[Union[Callable, Coroutine]], types.ModuleType]:
    """
    Dynamically import a Python file and return a list of functions it defines,
    and the module object.

    Args:
        filename (str): Path to the Python file to import

    Returns:
        tuple: (list of function objects, module object)
    """
    try:
        spec = importlib.util.spec_from_file_location("dynamic_module", filename)
        if spec is None:
            raise ImportError(f"Could not load spec for module: {filename}")

        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Could not load module: {filename}")

        spec.loader.exec_module(module)

        functions = []
        for name, obj in inspect.getmembers(module):
            # Filter for functions defined in the module
            if (
                inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)
            ) and obj.__module__ == module.__name__:
                functions.append(obj)

        return functions, module

    except Exception as e:
        raise ImportError(f"Error importing {filename}: {str(e)}")
