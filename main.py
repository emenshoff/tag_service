import os
import uvicorn
import logging
import argparse
import logging.config
import config
import yaml


'''
By default all settings are taken from root config.py file.
You can add optional arguments to run app
--host : overrewrite host name \ ip
--port : owerwrite port
--config : owerwrite json config path

'''

parser = argparse.ArgumentParser(description='Runtime app parameters...')

# Required positional arguments
parser.add_argument('--host', type=str, help='Service host name or ip address')
parser.add_argument('--port', type=int, help='Service port number')

# Optional positional argument
parser.add_argument('--config', type=str, nargs='?',help='Path to engine configuration .json file')


args = parser.parse_args()

_host = args.host
_port = args.port

_cfg_path = args.config

# overriting the default settings
if _cfg_path is not None:
    config.ENGINE_CONFIG_PATH = _cfg_path

if _host is not None:
    config.HOST = _host

if _port is not None:
    config.PORT = _port


# setting up logging
with open(config.LOGGING_CFG) as f:
    logging_cfg = yaml.load(f, Loader=yaml.FullLoader)

logging.config.dictConfig(logging_cfg)
log = logging.getLogger()



if __name__ == "__main__":    
    
    uvicorn.run(
                    "api.app:app", 
                    host=config.HOST, 
                    port=config.PORT, 
                    log_config=logging_cfg, 
                    reload=True
                )

