from components.utils.train import Train
from components.options import parse_arg

def main() -> None:
    cfg = parse_arg()
    trainer = Train(cfg=cfg)
    trainer.run_training()
    
if __name__ == "__main__":
    main()