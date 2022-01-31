import hydra

from config import Config
from runner import Runner
import matplotlib.pyplot as plt


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: Config):
    runner = Runner(cfg)
    runner.run_accuracy()
    # TODO: ImageNette dataset class. (Folders 0, 9 instead of folder names)

if __name__ == "__main__":
    main()
