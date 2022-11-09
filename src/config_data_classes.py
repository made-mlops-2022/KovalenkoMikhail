from dataclasses import dataclass


@dataclass()
class ConfigTrainPipeline:
    data_path: str
    target_col: str
    test_size: float
    model_filename: str
    penalty: str
    max_iter: int
    random_state: int


@dataclass()
class ConfigInferencePipeline:
    model_filename: str
