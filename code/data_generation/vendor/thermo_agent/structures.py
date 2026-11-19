from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"^(?P<material>[A-Za-z][A-Za-z0-9]*)_(?P<thickness>\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class Layer:
    material: str
    thickness_nm: float

    @property
    def token(self) -> str:
        thickness = int(self.thickness_nm) if self.thickness_nm.is_integer() else self.thickness_nm
        return f"{self.material}_{thickness}"


@dataclass(frozen=True)
class StructureValidation:
    valid: bool
    normalized: str
    structure_hash: str
    layers: tuple[Layer, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def load_vocab_tokens(path: str | Path) -> set[str]:
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def parse_structure(structure: str) -> tuple[Layer, ...]:
    layers = []
    for token in structure.split():
        match = TOKEN_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid material token: {token!r}")
        layers.append(Layer(match.group("material"), float(match.group("thickness"))))
    if not layers:
        raise ValueError("Structure is empty.")
    return tuple(layers)


def normalize_structure(layers: Iterable[Layer]) -> str:
    return " ".join(layer.token for layer in layers)


def validate_structure(
    structure: str,
    *,
    max_layers: int = 30,
    required_wvo2_count: int = 1,
    wvo2_min_thickness_nm: float = 20,
    wvo2_max_thickness_nm: float = 40,
    terminal_token: str = "SiO2_10000",
    allowed_tokens: set[str] | None = None,
) -> StructureValidation:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        layers = parse_structure(structure)
    except ValueError as exc:
        return StructureValidation(False, "", "", (), (str(exc),), ())

    normalized = normalize_structure(layers)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if len(layers) > max_layers:
        errors.append(f"layer_count={len(layers)} exceeds max_layers={max_layers}")
    if any(layer.thickness_nm <= 0 for layer in layers):
        errors.append("All layer thicknesses must be positive.")

    tokens = [layer.token for layer in layers]
    if tokens[-1] != terminal_token:
        errors.append(f"Last layer must be {terminal_token}.")
    if terminal_token in tokens[:-1]:
        errors.append(f"{terminal_token} may appear only as the last layer.")

    wvo2 = [layer for layer in layers if layer.material == "WVO2"]
    if len(wvo2) != required_wvo2_count:
        errors.append(f"Expected {required_wvo2_count} WVO2 layer(s), found {len(wvo2)}.")
    for layer in wvo2:
        if not wvo2_min_thickness_nm <= layer.thickness_nm <= wvo2_max_thickness_nm:
            errors.append(
                f"WVO2 thickness {layer.thickness_nm:g} nm is outside "
                f"[{wvo2_min_thickness_nm:g}, {wvo2_max_thickness_nm:g}] nm."
            )

    if allowed_tokens is not None:
        unknown = sorted(set(tokens).difference(allowed_tokens))
        if unknown:
            errors.append(f"Tokens absent from the frozen vocabulary: {unknown}")

    if normalized != structure.strip():
        warnings.append("Structure was normalized before hashing.")
    return StructureValidation(not errors, normalized, digest, layers, tuple(errors), tuple(warnings))
