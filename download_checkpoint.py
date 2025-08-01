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
    case 25:
        file_id='1aeWcuUV0jOs6S3PT-aHicjZ8sSTSOrEv'
    case 14:
        file_id='1c7juG16UViFy_etteX6yY1AGmPRIqJYY'
    case 21:
        file_id='1iBLUf1ag9eZ2chbbgd-m735-Fb-MWHQT'
    case 23:
        file_id='1kYp-w7zn5_nwYstUeh5wiy3-UkOS_wNG'
    case _:
        raise RuntimeError('Unknown run id')

gdown.download(f'https://drive.google.com/uc?id={file_id}', 'checkpoint/model.safetensors', quiet=False)



