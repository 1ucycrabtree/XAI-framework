from typing import Callable, Dict, Optional, Type


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Type] = {}

    def get(self, key: str) -> Type:
        cls = self._module_dict.get(key.lower())
        if cls is None:
            raise KeyError(
                f"{key} is not in the {self.name} registry. "
                f"Available keys: {list(self._module_dict.keys())}"
            )
        return cls

    def register_module(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class."""

        def _register(cls: Type) -> Type:
            key = name if name is not None else cls.__name__
            key = key.lower()
            if key in self._module_dict:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._module_dict[key] = cls
            return cls

        return _register
