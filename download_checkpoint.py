import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--run-id', type=int, required=True)
args = parser.parse_args()
import gdown

id = args.run_id
os.makedirs("checkpoint")

match id:
    case 24:
        file_id ='1zW-dDO_9m4MrgWRzs2s_U0dsF0sj8NOV'
    case _:
        raise RuntimeError('Unknown run id')

gdown.download(f'https://drive.google.com/uc?id={file_id}', 'checkpoint/model.safetensors', quiet=False)



