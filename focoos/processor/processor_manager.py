import importlib
from typing import Callable, Dict, Type

from focoos.ports import ModelConfig, ModelFamily
from focoos.processor.base_processor import BaseProcessor


class ProcessorManager:
    """Automatic processor manager with lazy loading"""

    _PROCESSOR_MAPPING: Dict[str, Callable[[], Type[BaseProcessor]]] = {}

    @classmethod
    def register_processor(cls, model_family: ModelFamily, processor_loader: Callable[[], Type[BaseProcessor]]):
        """
        Register a loader for a specific processor
        """
        cls._PROCESSOR_MAPPING[model_family.value] = processor_loader

    @classmethod
    def _ensure_family_registered(cls, model_family: ModelFamily):
        """Ensure the processor family is registered, importing if needed."""
        if model_family.value not in cls._PROCESSOR_MAPPING:
            family_module = importlib.import_module(f"focoos.models.{model_family.value}")
            for attr_name in dir(family_module):
                if attr_name.startswith("_register"):
                    register_func = getattr(family_module, attr_name)
                    if callable(register_func):
                        register_func()

    @classmethod
    def get_processor(cls, model_family: ModelFamily, model_config: ModelConfig) -> BaseProcessor:
        """
        Get a processor instance for the given model family.
        """
        cls._ensure_family_registered(model_family)
        if model_family.value not in cls._PROCESSOR_MAPPING:
            raise ValueError(f"Processor for {model_family} not supported")
        processor_class = cls._PROCESSOR_MAPPING[model_family.value]()
        return processor_class(config=model_config)
