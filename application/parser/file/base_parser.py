"""Base parser and config class."""

from abc import abstractmethod
from pathlib import Path


class BaseParser:
    """Base class for all parsers."""

    def __init__(self, parser_config: Optional[Dict] = None):
        """Init params."""
        self._parser_config = parser_config

    def init_parser(self) -> None:
        """Init parser and store it."""
        parser_config = self._init_parser()
        self._parser_config = parser_config

    @property
    def parser_config_set(self) -> bool:
        """Check if parser config is set."""
        return self._parser_config is not None

    @property
    @property
    def parser_config(self) -> dict:
        """Get the parser configuration."""
        if self._parser_config is None:
            raise ValueError("Parser config not set.")
        return self._parser_config

    @abstractmethod
    def _init_parser(self) -> Dict:
        """Initialize the parser with the config."""

    @abstractmethod
    def parse_file(self, file: Path, errors: str = "ignore") -> str | list[str]:
        """Parse file."""
