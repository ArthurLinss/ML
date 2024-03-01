import hydra
from omegaconf import DictConfig, OmegaConf
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """
    print config file settings stored in config file
    """
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
