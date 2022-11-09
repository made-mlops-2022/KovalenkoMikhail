import click
import yaml

from src.utils import inference
from src.config_data_classes import ConfigInferencePipeline


@click.command()
@click.argument('config_path')
def run_pipeline(config_path):
    with open(config_path, 'r') as config_file:
        config = ConfigInferencePipeline(**yaml.safe_load(config_file))

    inference(config.model_filename)


if __name__ == '__main__':
    run_pipeline()
