import hydra
from hydra.core.config_store import ConfigStore

from config import (
    AttackConfig,
    CIFAR10Config,
    CIFAR10GoogLeNetConfig,
    Config,
    ImageNetteConfig,
    ImageNetteGoogLeNetConfig,
)
from runner import Runner

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="dataset", name="base_cifar10", node=CIFAR10Config)
cs.store(group="dataset", name="base_imagenette", node=ImageNetteConfig)
cs.store(group="model", name="base_imagenettegooglenet", node=ImageNetteGoogLeNetConfig)
cs.store(group="model", name="base_cifar10googlenet", node=CIFAR10GoogLeNetConfig)
cs.store(group="attack", name="base_attack", node=AttackConfig)


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.mode.name == "accuracy":
        runner.run_accuracy()

    if cfg.mode.name == "normal_run":
        results_folder = (
            cfg.mode.results_path + cfg.dataset.name + "/" + cfg.model.name + "/"
        )
        results_file = results_folder + cfg.mode.noise.name + ".pt"
        import os

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        runner.run(results_file, cfg.mode.noise.standard_deviations)

    if cfg.mode.name == "attack_run":
        results_folder = (
            cfg.mode.results_path
            + cfg.dataset.name
            + "/"
            + cfg.model.name
            + "/"
            + cfg.mode.attack.name
            + "/"
        )
        results_file = results_folder + cfg.mode.noise.name + ".pt"
        from attack import Attack

        a = Attack(
            cfg.mode.attack,
            runner.loader.dataset.real_classes
            if bool(cfg.mode.attack.targeted)
            else None,
        )
        import os

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        runner.run_with_attack(results_file, cfg.mode.noise.standard_deviations, a)


if __name__ == "__main__":
    main()
