import hydra
#from omegaconf import OmegaConf

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    #OmegaConf.resolve(cfg)
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()

