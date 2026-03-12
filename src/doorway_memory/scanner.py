"""Data source scanning — automatic shape extraction.

Scans external data sources and converts them into shapes for
geometric memory. Supports five scan types:

- dataframe: pandas/numpy tabular data
- database: SQL table schemas and value ranges
- json: nested JSON structures
- openapi: API endpoint parameter schemas
- codebase: Python source code via ast parsing

Plus a scan() auto-detect convenience function.

Optional deps: pandas (dataframe), sqlalchemy (database), pyyaml (openapi YAML).
"""

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .shape import Shape, Dimension

# Minimum unique values to treat a column as a real dimension
MIN_UNIQUE_VALUES = 3

# Boundary margin on scanned ranges (fraction of range)
SCANNER_MARGIN = 0.05

# Minimum range width to avoid zero-width dimensions
MIN_RANGE_WIDTH = 1e-10

# Numeric type annotations recognized by codebase scanner
_NUMERIC_ANNOTATIONS = {"int", "float", "integer", "number", "double", "decimal"}

# Default numeric range when type is known but bounds are not
_DEFAULT_RANGES = {
    "int": (0.0, 1000.0),
    "integer": (0.0, 1000.0),
    "float": (0.0, 1.0),
    "number": (0.0, 1.0),
    "double": (0.0, 1.0),
    "decimal": (0.0, 1.0),
}


@dataclass
class ScanResult:
    """Result of a scan operation."""
    source: str
    source_type: str
    shapes: List[Shape] = field(default_factory=list)


def _add_margin(min_val: float, max_val: float, margin: float = SCANNER_MARGIN) -> Tuple[float, float]:
    """Add margin to a min/max range."""
    span = max_val - min_val
    if span < MIN_RANGE_WIDTH:
        span = MIN_RANGE_WIDTH
    return min_val - span * margin, max_val + span * margin


def _extract_numeric_fields(data: Dict, prefix: str = "") -> Dict[str, List[float]]:
    """Recursively extract numeric fields from a nested dict, grouped by dot-path key."""
    fields: Dict[str, List[float]] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields.setdefault(path, []).append(float(value))
            elif isinstance(value, dict):
                sub = _extract_numeric_fields(value, path)
                for k, v in sub.items():
                    fields.setdefault(k, []).extend(v)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        fields.setdefault(path, []).append(float(item))
                    elif isinstance(item, dict):
                        sub = _extract_numeric_fields(item, path)
                        for k, v in sub.items():
                            fields.setdefault(k, []).extend(v)
    return fields


def _schema_to_dimension(
    name: str,
    schema: Dict,
    margin: float = SCANNER_MARGIN,
) -> Optional[Dimension]:
    """Convert an OpenAPI/JSON-Schema field to a Dimension if numeric with bounds."""
    field_type = schema.get("type", "")
    if field_type not in ("number", "integer"):
        return None
    field_min = schema.get("minimum")
    field_max = schema.get("maximum")
    if field_min is not None and field_max is not None:
        lo, hi = _add_margin(float(field_min), float(field_max), margin)
        return Dimension(name=name, min_value=lo, max_value=hi)
    # Use defaults if no constraints
    defaults = _DEFAULT_RANGES.get(field_type)
    if defaults:
        lo, hi = _add_margin(defaults[0], defaults[1], margin)
        return Dimension(name=name, min_value=lo, max_value=hi)
    return None


def _annotation_to_dimension(
    name: str,
    annotation: str,
    margin: float = SCANNER_MARGIN,
) -> Optional[Dimension]:
    """Convert a Python type annotation string to a Dimension if numeric."""
    ann_lower = annotation.lower().strip()
    if ann_lower not in _NUMERIC_ANNOTATIONS:
        return None
    defaults = _DEFAULT_RANGES.get(ann_lower, (0.0, 1.0))
    lo, hi = _add_margin(defaults[0], defaults[1], margin)
    return Dimension(name=name, min_value=lo, max_value=hi)


class Scanner:
    """
    Scans external data sources and extracts geometric shapes.

    Configurable thresholds:
        min_unique_values: minimum unique values for a column to become a dimension
        margin: boundary margin fraction added to observed ranges
    """

    def __init__(
        self,
        min_unique_values: int = MIN_UNIQUE_VALUES,
        margin: float = SCANNER_MARGIN,
    ):
        self.min_unique_values = min_unique_values
        self.margin = margin

    def scan_dataframe(self, df: Any, name: str = "dataframe") -> ScanResult:
        """
        Scan a pandas DataFrame or dict-of-lists.

        Numeric columns with enough unique values become dimensions.
        One shape per source covering all qualifying columns.
        """
        if hasattr(df, 'columns') and hasattr(df, 'dtypes'):
            columns = {}
            for col in df.columns:
                try:
                    vals = df[col].dropna().astype(float).values
                    unique = len(np.unique(vals))
                    if unique >= self.min_unique_values:
                        columns[col] = vals
                except (ValueError, TypeError):
                    continue
        elif isinstance(df, dict):
            columns = {}
            for col, values in df.items():
                try:
                    vals = np.array([v for v in values if v is not None], dtype=float)
                    unique = len(np.unique(vals))
                    if unique >= self.min_unique_values:
                        columns[col] = vals
                except (ValueError, TypeError):
                    continue
        else:
            return ScanResult(source=name, source_type="dataframe")

        if not columns:
            return ScanResult(source=name, source_type="dataframe")

        dims = {}
        for col, vals in columns.items():
            lo, hi = _add_margin(float(np.min(vals)), float(np.max(vals)), self.margin)
            dims[col] = Dimension(name=col, min_value=lo, max_value=hi)

        shape = Shape(
            dimensions=dims,
            metadata={"scan_type": "dataframe", "source": name, "column_count": len(dims)},
        )
        return ScanResult(source=name, source_type="dataframe", shapes=[shape])

    def scan_database(self, schema: Dict[str, Dict], name: str = "database") -> ScanResult:
        """
        Scan a database schema definition.

        Args:
            schema: Dict mapping table names to column defs.
                Column defs have "type", optional "min"/"max".
            name: Source identifier.
        """
        numeric_types = {"numeric", "integer", "float", "double", "decimal", "real", "int"}
        shapes = []

        for table_name, columns in schema.items():
            dims = {}
            for col_name, col_def in columns.items():
                col_type = col_def.get("type", "").lower()
                if col_type not in numeric_types:
                    continue
                col_min = col_def.get("min")
                col_max = col_def.get("max")
                if col_min is not None and col_max is not None:
                    lo, hi = _add_margin(float(col_min), float(col_max), self.margin)
                    dims[col_name] = Dimension(name=col_name, min_value=lo, max_value=hi)

            if dims:
                shapes.append(Shape(
                    dimensions=dims,
                    metadata={"scan_type": "database", "source": name, "table": table_name},
                ))

        return ScanResult(source=name, source_type="database", shapes=shapes)

    def scan_json(self, data: Any, name: str = "json") -> ScanResult:
        """
        Scan a JSON structure. Nested fields use dot-path names.

        Lists of dicts extract ranges across all records.
        """
        if isinstance(data, list):
            # List of dicts — merge numeric fields across records
            merged: Dict[str, List[float]] = {}
            for item in data:
                if isinstance(item, dict):
                    fields = _extract_numeric_fields(item)
                    for k, v in fields.items():
                        merged.setdefault(k, []).extend(v)
            fields_to_use = merged
        elif isinstance(data, dict):
            fields_to_use = _extract_numeric_fields(data)
        else:
            return ScanResult(source=name, source_type="json")

        dims = {}
        for path, values in fields_to_use.items():
            if not values:
                continue
            lo, hi = _add_margin(min(values), max(values), self.margin)
            dims[path] = Dimension(name=path, min_value=lo, max_value=hi)

        if not dims:
            return ScanResult(source=name, source_type="json")

        shape = Shape(
            dimensions=dims,
            metadata={"scan_type": "json", "source": name},
        )
        return ScanResult(source=name, source_type="json", shapes=[shape])

    def scan_openapi(self, spec: Dict, name: str = "openapi") -> ScanResult:
        """
        Scan an OpenAPI specification.

        One shape per endpoint with numeric parameters that have min/max
        constraints. Endpoints without constraints use default ranges.
        """
        shapes = []
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            for method, details in methods.items():
                if not isinstance(details, dict):
                    continue
                dims = {}

                for param in details.get("parameters", []):
                    param_name = param.get("name", "")
                    schema = param.get("schema", {})
                    dim = _schema_to_dimension(param_name, schema, self.margin)
                    if dim:
                        dims[param_name] = dim

                request_body = details.get("requestBody", {})
                content = request_body.get("content", {})
                for content_type, content_def in content.items():
                    schema = content_def.get("schema", {})
                    for prop_name, prop_schema in schema.get("properties", {}).items():
                        dim = _schema_to_dimension(prop_name, prop_schema, self.margin)
                        if dim:
                            dims[prop_name] = dim

                if dims:
                    shapes.append(Shape(
                        dimensions=dims,
                        metadata={
                            "scan_type": "openapi",
                            "source": name,
                            "path": path,
                            "method": method.upper(),
                        },
                    ))

        return ScanResult(source=name, source_type="openapi", shapes=shapes)

    def scan_codebase(self, path: str, name: str = "codebase") -> ScanResult:
        """
        Scan Python source files using ast parsing.

        Extracts typed function parameters (int/float annotations)
        and converts them to dimensions. One shape per function
        that has numeric typed parameters.
        """
        p = Path(path)
        shapes = []

        if p.is_file() and p.suffix == ".py":
            py_files = [p]
        elif p.is_dir():
            py_files = list(p.rglob("*.py"))
        else:
            return ScanResult(source=name, source_type="codebase")

        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                dims = {}
                for arg in node.args.args:
                    if arg.annotation is None:
                        continue
                    ann_str = _get_annotation_string(arg.annotation)
                    if ann_str:
                        dim = _annotation_to_dimension(arg.arg, ann_str, self.margin)
                        if dim:
                            dims[arg.arg] = dim

                if dims:
                    func_name = f"{py_file.stem}.{node.name}"
                    shapes.append(Shape(
                        dimensions=dims,
                        metadata={
                            "scan_type": "codebase",
                            "source": name,
                            "module": py_file.stem,
                            "function": node.name,
                        },
                    ))

        return ScanResult(source=name, source_type="codebase", shapes=shapes)


def _get_annotation_string(node: ast.AST) -> Optional[str]:
    """Extract type annotation as a string from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def scan(source: Any, name: str = "auto", metadata: Optional[Dict] = None) -> ScanResult:
    """
    Auto-detect source type and scan.

    Detection rules:
        - object with .columns and .dtypes → dataframe
        - str path to .py file or directory → codebase
        - dict with "paths" key → openapi
        - dict with table→columns structure (with "type") → database
        - list of dicts → json
        - dict → json
    """
    scanner = Scanner()

    # DataFrame
    if hasattr(source, 'columns') and hasattr(source, 'dtypes'):
        return scanner.scan_dataframe(source, name)

    # Path to codebase
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and (p.is_dir() or p.suffix == ".py"):
            return scanner.scan_codebase(str(p), name)

    if isinstance(source, dict):
        # OpenAPI
        if "paths" in source:
            return scanner.scan_openapi(source, name)

        # Database schema: values are dicts whose values have "type"
        first_val = next(iter(source.values()), None) if source else None
        if isinstance(first_val, dict):
            inner_first = next(iter(first_val.values()), None) if first_val else None
            if isinstance(inner_first, dict) and "type" in inner_first:
                return scanner.scan_database(source, name)

        # Dict-of-lists → dataframe
        if isinstance(first_val, list):
            return scanner.scan_dataframe(source, name)

    # List of dicts → json
    if isinstance(source, list):
        return scanner.scan_json(source, name)

    # Fallback: JSON
    if isinstance(source, dict):
        return scanner.scan_json(source, name)

    return ScanResult(source=name, source_type="unknown")
