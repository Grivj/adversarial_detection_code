import hydra

from config import Config
from utils import get_fmodel, get_loader


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: Config):
    loader = get_loader(cfg.dataset)
    model = get_fmodel(cfg.model)
    x, y = next(iter(loader))
    print(model(x))


if __name__ == "__main__":
    main()
