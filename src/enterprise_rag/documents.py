from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# It is data class that means Python automatically gives it: __init__,readable fields, simple object behavior
@dataclass(frozen=True)
class CanonicalDocument:
    document_id: str
    source: str
    source_id: str
    title: str
    version: str
    updated_at: str
    text: str
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_json(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{self.document_id}.json"
        file_path.write_text(self.to_json(), encoding="utf-8")

        return file_path