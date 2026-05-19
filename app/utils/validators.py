from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Iterable, List
import pandas as pd
from io import StringIO


def normalize_column_name(name: str) -> str:
    return ''.join(ch for ch in str(name).strip().lower() if ch.isalnum())


def read_csv_flexible(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text), sep=None, engine='python')
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}")


def map_dataframe_columns(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    normalized_columns = {normalize_column_name(col): col for col in df.columns}
    rename_map: Dict[str, str] = {}
    missing: List[str] = []

    for expected in expected_columns:
        key = normalize_column_name(expected)
        if key in normalized_columns:
            rename_map[normalized_columns[key]] = expected
        else:
            missing.append(expected)

    if missing:
        raise ValueError(f"CSV must contain columns: {expected_columns}. Found: {list(df.columns)}")

    return df.rename(columns=rename_map)


class OccupancyFeatures(BaseModel):
    temperature: float
    humidity: float
    light: float
    co2: float
    humidity_ratio: float

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < -50 or v > 100:
            raise ValueError('Temperature must be between -50 and 100')
        return v

    @field_validator('humidity')
    @classmethod
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Humidity must be between 0 and 100')
        return v

    @field_validator('light')
    @classmethod
    def validate_light(cls, v):
        if v < 0:
            raise ValueError('Light must be non-negative')
        return v

    @field_validator('co2')
    @classmethod
    def validate_co2(cls, v):
        if v < 0:
            raise ValueError('CO2 must be non-negative')
        return v

    @field_validator('humidity_ratio')
    @classmethod
    def validate_humidity_ratio(cls, v):
        if v < 0:
            raise ValueError('Humidity ratio must be non-negative')
        return v