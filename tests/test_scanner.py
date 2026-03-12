"""Tests for scanner.py — automatic shape extraction."""

import os
import tempfile
from pathlib import Path

from doorway_memory.shape import Dimension, Shape
from doorway_memory.scanner import (
    Scanner,
    ScanResult,
    scan,
    _extract_numeric_fields,
    _schema_to_dimension,
    _annotation_to_dimension,
    MIN_UNIQUE_VALUES,
    SCANNER_MARGIN,
)
from doorway_memory.memory import Memory


# --- ScanResult ---

def test_scan_result_dataclass():
    r = ScanResult(source="test", source_type="json")
    assert r.source == "test"
    assert r.source_type == "json"
    assert r.shapes == []


# --- scan_dataframe ---

def test_scan_dataframe_numeric_columns():
    scanner = Scanner()
    data = {
        "temperature": [20.0, 25.0, 30.0, 35.0],
        "pressure": [1.0, 1.1, 1.2, 0.9],
    }
    result = scanner.scan_dataframe(data, "sensors")
    assert len(result.shapes) == 1
    shape = result.shapes[0]
    assert "temperature" in shape.dimensions
    assert "pressure" in shape.dimensions
    # Ranges should cover observed min/max plus margin
    assert shape.dimensions["temperature"].min_value < 20.0
    assert shape.dimensions["temperature"].max_value > 35.0


def test_scan_dataframe_skips_non_numeric():
    scanner = Scanner()
    data = {
        "name": ["alice", "bob", "charlie"],
        "score": [80.0, 90.0, 100.0],
    }
    result = scanner.scan_dataframe(data, "test")
    assert len(result.shapes) == 1
    shape = result.shapes[0]
    assert "name" not in shape.dimensions
    assert "score" in shape.dimensions


def test_scan_dataframe_skips_low_cardinality():
    scanner = Scanner(min_unique_values=3)
    data = {
        "flag": [0.0, 1.0, 0.0, 1.0],  # only 2 unique values
        "value": [10.0, 20.0, 30.0, 40.0],  # 4 unique values
    }
    result = scanner.scan_dataframe(data, "test")
    assert len(result.shapes) == 1
    assert "flag" not in result.shapes[0].dimensions
    assert "value" in result.shapes[0].dimensions


def test_scan_dataframe_empty():
    scanner = Scanner()
    result = scanner.scan_dataframe({}, "empty")
    assert result.shapes == []


def test_scan_dataframe_metadata():
    scanner = Scanner()
    data = {"x": [1.0, 2.0, 3.0]}
    result = scanner.scan_dataframe(data, "mydata")
    assert result.source == "mydata"
    assert result.source_type == "dataframe"
    assert result.shapes[0].metadata["scan_type"] == "dataframe"


# --- scan_database ---

def test_scan_database_numeric_columns():
    scanner = Scanner()
    schema = {
        "users": {
            "age": {"type": "integer", "min": 0, "max": 150},
            "score": {"type": "float", "min": 0.0, "max": 100.0},
            "name": {"type": "text"},
        }
    }
    result = scanner.scan_database(schema, "mydb")
    assert len(result.shapes) == 1
    shape = result.shapes[0]
    assert "age" in shape.dimensions
    assert "score" in shape.dimensions
    assert "name" not in shape.dimensions


def test_scan_database_multiple_tables():
    scanner = Scanner()
    schema = {
        "orders": {
            "amount": {"type": "decimal", "min": 0, "max": 10000},
        },
        "products": {
            "price": {"type": "float", "min": 0.0, "max": 500.0},
        },
    }
    result = scanner.scan_database(schema, "shop")
    assert len(result.shapes) == 2


def test_scan_database_no_numeric():
    scanner = Scanner()
    schema = {"logs": {"message": {"type": "text"}}}
    result = scanner.scan_database(schema, "logs")
    assert result.shapes == []


def test_scan_database_metadata():
    scanner = Scanner()
    schema = {"t1": {"x": {"type": "integer", "min": 0, "max": 10}}}
    result = scanner.scan_database(schema, "db")
    assert result.shapes[0].metadata["table"] == "t1"
    assert result.shapes[0].metadata["scan_type"] == "database"


# --- scan_json ---

def test_scan_json_flat():
    scanner = Scanner()
    data = {"temperature": 25.0, "humidity": 60.0}
    result = scanner.scan_json(data, "sensor_reading")
    assert len(result.shapes) == 1
    assert "temperature" in result.shapes[0].dimensions
    assert "humidity" in result.shapes[0].dimensions


def test_scan_json_nested():
    scanner = Scanner()
    data = {"sensor": {"temp": 25.0, "pressure": 1.0}}
    result = scanner.scan_json(data, "nested")
    assert len(result.shapes) == 1
    assert "sensor.temp" in result.shapes[0].dimensions
    assert "sensor.pressure" in result.shapes[0].dimensions


def test_scan_json_list_of_dicts():
    scanner = Scanner()
    data = [
        {"temp": 20.0, "humidity": 50.0},
        {"temp": 30.0, "humidity": 70.0},
        {"temp": 25.0, "humidity": 60.0},
    ]
    result = scanner.scan_json(data, "records")
    assert len(result.shapes) == 1
    shape = result.shapes[0]
    assert "temp" in shape.dimensions
    # Range should span observed values
    assert shape.dimensions["temp"].min_value < 20.0
    assert shape.dimensions["temp"].max_value > 30.0


def test_scan_json_no_numeric():
    scanner = Scanner()
    data = {"name": "test", "active": True}
    result = scanner.scan_json(data, "strings")
    assert result.shapes == []


def test_scan_json_metadata():
    scanner = Scanner()
    result = scanner.scan_json({"x": 1.0}, "mydata")
    assert result.source_type == "json"
    assert result.shapes[0].metadata["scan_type"] == "json"


# --- scan_openapi ---

def test_scan_openapi_with_constraints():
    scanner = Scanner()
    spec = {
        "paths": {
            "/users": {
                "get": {
                    "parameters": [
                        {
                            "name": "age",
                            "schema": {"type": "integer", "minimum": 0, "maximum": 150},
                        },
                        {
                            "name": "score",
                            "schema": {"type": "number", "minimum": 0.0, "maximum": 100.0},
                        },
                    ]
                }
            }
        }
    }
    result = scanner.scan_openapi(spec, "api")
    assert len(result.shapes) == 1
    shape = result.shapes[0]
    assert "age" in shape.dimensions
    assert "score" in shape.dimensions
    assert shape.metadata["path"] == "/users"
    assert shape.metadata["method"] == "GET"


def test_scan_openapi_without_constraints():
    scanner = Scanner()
    spec = {
        "paths": {
            "/data": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "properties": {
                                        "value": {"type": "number"},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    result = scanner.scan_openapi(spec, "api")
    assert len(result.shapes) == 1
    # Should use default range for "number"
    assert "value" in result.shapes[0].dimensions


def test_scan_openapi_no_numeric_params():
    scanner = Scanner()
    spec = {
        "paths": {
            "/search": {
                "get": {
                    "parameters": [
                        {"name": "q", "schema": {"type": "string"}},
                    ]
                }
            }
        }
    }
    result = scanner.scan_openapi(spec, "api")
    assert result.shapes == []


def test_scan_openapi_multiple_endpoints():
    scanner = Scanner()
    spec = {
        "paths": {
            "/a": {"get": {"parameters": [
                {"name": "x", "schema": {"type": "integer", "minimum": 0, "maximum": 10}},
            ]}},
            "/b": {"post": {"parameters": [
                {"name": "y", "schema": {"type": "number", "minimum": 0, "maximum": 1}},
            ]}},
        }
    }
    result = scanner.scan_openapi(spec, "api")
    assert len(result.shapes) == 2


# --- scan_codebase ---

def test_scan_codebase_typed_function():
    scanner = Scanner()
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "example.py"
        py_file.write_text(
            "def process(x: int, y: float, name: str) -> float:\n"
            "    return x + y\n"
        )
        result = scanner.scan_codebase(tmpdir, "mycode")
    assert len(result.shapes) == 1
    shape = result.shapes[0]
    assert "x" in shape.dimensions
    assert "y" in shape.dimensions
    # "name" is str, should be excluded
    assert "name" not in shape.dimensions


def test_scan_codebase_no_types():
    scanner = Scanner()
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "untyped.py"
        py_file.write_text(
            "def process(x, y, name):\n"
            "    return x + y\n"
        )
        result = scanner.scan_codebase(tmpdir, "untyped")
    assert result.shapes == []


def test_scan_codebase_multiple_functions():
    scanner = Scanner()
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "multi.py"
        py_file.write_text(
            "def foo(a: int) -> int:\n"
            "    return a\n\n"
            "def bar(b: float) -> float:\n"
            "    return b\n"
        )
        result = scanner.scan_codebase(tmpdir, "multi")
    assert len(result.shapes) == 2


def test_scan_codebase_single_file():
    scanner = Scanner()
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "single.py"
        py_file.write_text("def calc(x: int) -> int:\n    return x\n")
        result = scanner.scan_codebase(str(py_file), "single")
    assert len(result.shapes) == 1


def test_scan_codebase_metadata():
    scanner = Scanner()
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "mod.py"
        py_file.write_text("def fn(x: float) -> float:\n    return x\n")
        result = scanner.scan_codebase(tmpdir, "mycode")
    assert result.shapes[0].metadata["scan_type"] == "codebase"
    assert result.shapes[0].metadata["function"] == "fn"


def test_scan_codebase_nonexistent():
    scanner = Scanner()
    result = scanner.scan_codebase("/nonexistent/path", "missing")
    assert result.shapes == []


# --- scan() auto-detect ---

def test_scan_auto_detect_dataframe():
    data = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
    result = scan(data)
    assert result.source_type == "dataframe"
    assert len(result.shapes) == 1


def test_scan_auto_detect_dict_json():
    data = {"temperature": 25.0, "pressure": 1.0}
    result = scan(data)
    assert result.source_type == "json"
    assert len(result.shapes) == 1


def test_scan_auto_detect_openapi():
    spec = {"paths": {"/test": {"get": {"parameters": [
        {"name": "x", "schema": {"type": "integer", "minimum": 0, "maximum": 10}},
    ]}}}}
    result = scan(spec)
    assert result.source_type == "openapi"


def test_scan_auto_detect_database():
    schema = {
        "users": {
            "age": {"type": "integer", "min": 0, "max": 100},
        }
    }
    result = scan(schema)
    assert result.source_type == "database"


def test_scan_auto_detect_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "sample.py"
        py_file.write_text("def fn(x: int) -> int:\n    return x\n")
        result = scan(tmpdir)
    assert result.source_type == "codebase"


def test_scan_auto_detect_list_of_dicts():
    data = [{"x": 1.0}, {"x": 2.0}]
    result = scan(data)
    assert result.source_type == "json"


# --- Helper functions ---

def test_extract_numeric_fields_flat():
    fields = _extract_numeric_fields({"a": 1.0, "b": "text", "c": 3})
    assert "a" in fields
    assert "c" in fields
    assert "b" not in fields


def test_extract_numeric_fields_nested():
    fields = _extract_numeric_fields({"outer": {"inner": 5.0}})
    assert "outer.inner" in fields


def test_schema_to_dimension_numeric():
    dim = _schema_to_dimension("x", {"type": "integer", "minimum": 0, "maximum": 100})
    assert dim is not None
    assert dim.min_value < 0
    assert dim.max_value > 100


def test_schema_to_dimension_string():
    dim = _schema_to_dimension("x", {"type": "string"})
    assert dim is None


def test_schema_to_dimension_no_bounds():
    dim = _schema_to_dimension("x", {"type": "number"})
    assert dim is not None  # Should use defaults


def test_annotation_to_dimension_int():
    dim = _annotation_to_dimension("x", "int")
    assert dim is not None


def test_annotation_to_dimension_float():
    dim = _annotation_to_dimension("x", "float")
    assert dim is not None


def test_annotation_to_dimension_str():
    dim = _annotation_to_dimension("x", "str")
    assert dim is None


# --- Memory.scan_and_store integration ---

def test_scan_and_store_integration():
    mem = Memory(anchor=False)
    data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]}
    count = mem.scan_and_store(data)
    assert count == 1
    assert mem.count() == 1
    # Point inside scanned ranges should be known
    assert mem.is_known({"x": 2.5, "y": 25.0}) is True
    # Point outside should be void
    assert mem.is_void({"x": 100.0, "y": 100.0}) is True


def test_scan_and_store_multiple_tables():
    mem = Memory(anchor=False)
    schema = {
        "orders": {"amount": {"type": "float", "min": 0, "max": 1000}},
        "products": {"price": {"type": "float", "min": 0, "max": 500}},
    }
    count = mem.scan_and_store(schema)
    assert count == 2
    assert mem.count() == 2


def test_scan_and_store_empty():
    mem = Memory(anchor=False)
    count = mem.scan_and_store({"name": "text only"})
    assert count == 0
    assert mem.count() == 0
