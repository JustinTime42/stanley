"""JSON/YAML file loader."""

import logging
import json
import yaml
from pathlib import Path
from typing import Optional, Any
import uuid

from ....models.document_models import Document, DocumentType

logger = logging.getLogger(__name__)


class JSONLoader:
    """
    Loader for JSON and YAML files.

    PATTERN: Parse and flatten structured data
    CRITICAL: Convert to readable text format
    """

    def __init__(self):
        """Initialize JSON loader."""
        self.logger = logger

    async def load_file(self, file_path: str) -> Optional[Document]:
        """
        Load a JSON or YAML file.

        Args:
            file_path: Path to JSON/YAML file

        Returns:
            Document with parsed content
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()

            # Read file content
            with open(path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Parse based on extension
            if extension == ".json":
                data = json.loads(raw_content)
                doc_type = DocumentType.JSON
            elif extension in {".yaml", ".yml"}:
                data = yaml.safe_load(raw_content)
                doc_type = DocumentType.YAML
            else:
                self.logger.error(f"Unsupported file type: {extension}")
                return None

            # Convert to readable text
            content = self._format_data(data)

            # Extract metadata
            metadata = {
                "file_path": str(path),
                "file_name": path.name,
                "extension": extension,
                "data_type": type(data).__name__,
            }

            # Add structural info
            if isinstance(data, dict):
                metadata["key_count"] = len(data)
                metadata["top_level_keys"] = list(data.keys())[
                    :20
                ]  # Limit for large files
            elif isinstance(data, list):
                metadata["item_count"] = len(data)

            # Create document
            document = Document(
                id=str(uuid.uuid4()),
                content=content,
                type=doc_type,
                source=str(path),
                metadata=metadata,
            )

            self.logger.info(f"Loaded {doc_type.value} file: {path}")
            return document

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            return None

    def _format_data(
        self,
        data: Any,
        prefix: str = "",
        max_depth: int = 10,
        current_depth: int = 0,
    ) -> str:
        """
        Format nested data structure as readable text.

        Args:
            data: Data to format
            prefix: Current indentation prefix
            max_depth: Maximum nesting depth
            current_depth: Current depth

        Returns:
            Formatted text representation
        """
        if current_depth >= max_depth:
            return f"{prefix}... (max depth reached)\n"

        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(
                        self._format_data(
                            value,
                            prefix + "  ",
                            max_depth,
                            current_depth + 1,
                        )
                    )
                else:
                    lines.append(f"{prefix}{key}: {value}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(
                        self._format_data(
                            item,
                            prefix + "  ",
                            max_depth,
                            current_depth + 1,
                        )
                    )
                else:
                    lines.append(f"{prefix}- {item}")

        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)

    def supports_file(self, file_path: str) -> bool:
        """
        Check if loader supports this file.

        Args:
            file_path: Path to file

        Returns:
            True if file is JSON or YAML
        """
        extension = Path(file_path).suffix.lower()
        return extension in {".json", ".yaml", ".yml"}
