from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


STV = Tuple[float, float]


@dataclass(frozen=True)
class PropertySpec:
    """A source-agnostic description of one property to extract."""

    name: str
    description: str
    agent: str
    value_type: str = "category"
    field_paths: Tuple[str, ...] = field(default_factory=tuple)
    allowed_values: Tuple[str, ...] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    include_in_metta: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PropertySpec":
        name = str(data.get("name", "")).strip()
        field_paths = data.get("field_paths") or data.get("fields") or []
        allowed_values = data.get("allowed_values") or data.get("values") or []
        if isinstance(field_paths, str):
            field_paths = [field_paths]
        if isinstance(allowed_values, str):
            allowed_values = [allowed_values]

        return cls(
            name=name,
            description=str(data.get("description", name)).strip() or name,
            agent=str(data.get("agent") or data.get("extractor") or "text_llm").strip(),
            value_type=str(data.get("value_type") or "category").strip(),
            field_paths=tuple(str(path).strip() for path in field_paths if str(path).strip()),
            allowed_values=tuple(str(value).strip() for value in allowed_values if str(value).strip()),
            parameters=dict(data.get("parameters") or data.get("config") or {}),
            include_in_metta=bool(data.get("include_in_metta", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "agent": self.agent,
            "value_type": self.value_type,
            "field_paths": list(self.field_paths),
            "allowed_values": list(self.allowed_values),
            "parameters": dict(self.parameters),
            "include_in_metta": self.include_in_metta,
        }


@dataclass(frozen=True)
class ExtractionPlan:
    """The contract between the planner and specialized extractor agents."""

    source_name: str
    entity_type: str
    id_fields: Tuple[str, ...]
    text_fields: Tuple[str, ...]
    properties: Tuple[PropertySpec, ...]
    version: int = 1
    planner: str = "unknown"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source_name: str = "json") -> "ExtractionPlan":
        properties = data.get("properties") or data.get("property_specs") or []
        id_fields = data.get("id_fields") or data.get("identity_fields") or data.get("id_field") or ["id"]
        text_fields = data.get("text_fields") or []
        if isinstance(id_fields, str):
            id_fields = [id_fields]
        if isinstance(text_fields, str):
            text_fields = [text_fields]

        return cls(
            source_name=str(data.get("source_name") or source_name),
            entity_type=str(data.get("entity_type") or "item"),
            id_fields=tuple(str(field).strip() for field in id_fields if str(field).strip()),
            text_fields=tuple(str(field).strip() for field in text_fields if str(field).strip()),
            properties=tuple(PropertySpec.from_dict(item) for item in properties),
            version=int(data.get("version") or 1),
            planner=str(data.get("planner") or "llm"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "planner": self.planner,
            "source_name": self.source_name,
            "entity_type": self.entity_type,
            "id_fields": list(self.id_fields),
            "text_fields": list(self.text_fields),
            "properties": [spec.to_dict() for spec in self.properties],
        }

    @property
    def metta_properties(self) -> Sequence[PropertySpec]:
        return tuple(spec for spec in self.properties if spec.include_in_metta)


@dataclass(frozen=True)
class SourceRecord:
    source_name: str
    source_id: str
    raw: Mapping[str, Any]
    title: str = ""
    text: str = ""
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class AgentResult:
    property_name: str
    value: Any
    stv: STV
    evidence: str = ""
    source_agent: str = ""

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "stv": self.stv,
            "evidence": self.evidence,
            "source_agent": self.source_agent,
        }
