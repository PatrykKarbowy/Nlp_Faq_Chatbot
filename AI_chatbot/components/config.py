from pathlib import Path
from typing import Union

from pydantic import validator, BaseModel


class DLTrainingConfig(BaseModel):
    # Data parameters
    batch_size: int
    
    # Training parameters
    learning_rate: float
    max_epochs: int
    loss: str
    device: str
    
    # Model parameters
    hidden_layers: int
    
    # Optimizer parameters
    weight_decay: int
    optimizer: str
    
    # Datamodule parameters
    data_dir: Path
    
    @validator("data_dir")
    def validate_data_path(cls, v: Union[str, Path]) -> Path:
        return Path(v).resolve()
    
    @validator("batch_size", "max_epochs", "hidden_layers", "weight_decay", "device")
    def validate_str_field(cls, v: Union[str, int, float]) -> int:
        if isinstance(v, int):
            return v
        return abs(int(float(v)))
    
    @validator("learning_rate")
    def validate_float_field(cls, v: Union[str, int, float]) -> int:
        if isinstance(v, float):
            return v
        return abs(float(v))
    
    @validator("loss", "optimizer")
    def validate_string_field(cls, v: Union[str, int, float]) -> int:
        if not isinstance(v, str):
            raise ValueError("Value must be a string")
        return v