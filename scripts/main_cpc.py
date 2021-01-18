import os, sys

# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.benchmark import *
import configparser
from models.utils import *
from models.networkSwitch import *
import argparse


torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--save-every', default=None, type=int,help='save model per how many epochs')

parser.add_argument('--resume', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether resume training/validation from checkpoint')

parser.add_argument('--ini-file', default=None, required=True,
                    help='the path to the setting9.ini')

args = parser.parse_args()
args_json = get_config_dic(args.ini_file)
print(args_json)
model_type = get_model_type(args,args_json)
model=eval(model_type)
if model_type == 'CDCK2':
    assert args_json['conv_sizes'][-1] % args_json['n_frames'] == 0  # THe final output channels should be multiples of the n_frames

args_json['top_path'] = script_dir
args_json['scr_data_path'] = data_dir
args_json['logging_dir'] = os.path.join(top_path, 'logs', 'cpc', model_type)
args_json['setting_name']=get_setting_name(os.path.split(args.ini_file)[1])
args_json['run_name'] = args_json['setting_name'] + time.strftime("-%Y-%m-%d_%H_%M_%S")
args_json['model_type']=model_type
args_json['resume'] = args.resume
args_json['save_every']=args.save_every
args_json['ini_file']=args.ini_file
cpc_main(model, args_json)
