from collections.abc import Callable
from typing import Any


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, Callable[..., Any]]] = {}

    def register(self, name: str, description: str, function: Callable[..., Any]) -> None:
        if name in self._tools:
            raise ValueError(f"Tool is already registered: {name}")
        self._tools[name] = (description, function)

    def run(self, name: str, **arguments: Any) -> Any:
        try:
            _, function = self._tools[name]
        except KeyError as error:
            raise ValueError(f"Unknown monitoring tool: {name}") from error
        return function(**arguments)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": description}
            for name, (description, _) in self._tools.items()
        ]