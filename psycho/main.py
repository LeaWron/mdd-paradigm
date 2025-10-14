import hydra
from omegaconf import DictConfig

from psycho.session import run_session as session_entry


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    session_entry(cfg)


if __name__ == "__main__":
    main()
