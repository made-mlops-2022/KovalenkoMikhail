import pandas as pd
import click
import yaml

from src.utils import split_data, train_log_reg
from src.config_data_classes import ConfigTrainPipeline


@click.command()
@click.argument('config_path')
def run_pipeline(config_path):
    with open(config_path, 'r') as config_file:
        config = ConfigTrainPipeline(**yaml.safe_load(config_file))

    train_pipeline(
        data_path=config.data_path,
        target_col=config.target_col,
        test_size=config.test_size,
        model_filename=config.model_filename,
        penalty=config.penalty,
        max_iter=config.max_iter,
        random_state=config.random_state
    )


def train_pipeline(data_path, target_col='condition', test_size=0.2, model_filename='model.pkl',
                   penalty='l2', max_iter=1000, random_state=42):
    data = pd.read_csv(data_path)
    split_data(data, target_col, test_size=test_size, random_state=random_state)
    train_log_reg(
        filename=model_filename,
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state
    )


if __name__ == '__main__':
    run_pipeline()
