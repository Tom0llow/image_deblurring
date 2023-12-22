import os
import sys
from hydra import compose, initialize_config_dir


class Config:
    @staticmethod
    def get_conf():
        conf_dir = os.path.join(os.getcwd(), "conf")
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
            cnf = compose(config_name="config.yaml")
            return cnf
