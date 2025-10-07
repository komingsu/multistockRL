from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import PRIMARY_KEY, get_schema, validate_dataframe
from src.utils.config import Config, load_config


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config("configs/base.yaml")


@pytest.fixture(scope="session")
def dataset(config: Config) -> pd.DataFrame:
    return pd.read_csv(config.data_path)


def test_processed_csv_matches_schema(dataset: pd.DataFrame) -> None:
    errors = validate_dataframe(dataset)
    assert errors == [], f"Schema violations detected: {errors}"


def test_primary_key_is_unique(dataset: pd.DataFrame) -> None:
    duplicates = int(dataset.duplicated(subset=list(PRIMARY_KEY)).sum())
    assert duplicates == 0, f"Primary key duplicates found: {duplicates}"

def test_feature_columns_are_known(config: Config) -> None:
    schema_names = {spec.name for spec in get_schema()}
    feature_columns: list[str] = []
    columns = config.data.get("feature_columns", {})
    if isinstance(columns, dict):
        for values in columns.values():
            feature_columns.extend(values)
    else:
        feature_columns.extend(columns)
    feature_columns.append(config.data["target_column"])

    missing = sorted(set(feature_columns) - schema_names)
    assert not missing, f"Config references unknown feature columns: {missing}"
