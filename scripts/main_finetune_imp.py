import os, sys

# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import configparser
from models.utils import *
from models.downstream import *
from models.networks import *
from models.benchmark import *
from models.networkSwitch import CPAE1_S,CPAE4_S,CPLSTM2,CAE_LSTM,AE_LSTM,CAE2_S, CDCK3_S, CPAELSTM44_AT
from joblib import Parallel, delayed


torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model-seed', default=None, type=int)

parser.add_argument('--ini-file', default=None, required=True,
                    help='the path to the setting9.ini')
parser.add_argument('--resume', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--freeze', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--mode', default=None, required=False,
                    help='whether to just test percentage at 0.05,0.001')

args = parser.parse_args()

print('===========================================')
print('==========imp=============')
print('Fine-tune according to %s' % args.ini_file)
print('Freeze = %s ' % args.freeze)
print(args.ini_file)
clf_args = get_config_dic(args.ini_file)
print(clf_args)
assert clf_args is not None
print (clf_args)
try:cpc_args_file = os.path.join(top_path, clf_args['model_setting'])
except KeyError: print('ini file seems wrong: %s'%args.ini_file)
model_type = get_model_type(cpc_args_file)
model = eval(model_type)
cpc_args = get_config_dic(cpc_args_file)
print(cpc_args)
if model_type == 'CDCK2':
    assert cpc_args['conv_sizes'][-1] % cpc_args[
        'n_frames'] == 0  # THe final output channels should be multiples of the n_frames
if args.model_seed is not None:
    clf_args['model_seed']=args.model_seed
if args.model_seed is None and clf_args.get('model_seed') is None:
    clf_args['model_seed'] = 0
clf_args['resume']=args.resume
cpc_args['learning_rate'] = None
cpc_args['model_type'] = model_type
cpc_args['top_path'] = script_dir
cpc_args['scr_data_path'] = data_dir
clf_args['setting_name'] = 'EarlyStopping15_'+get_setting_name(os.path.split(args.ini_file)[1])
clf_args['run_name'] = cpc_args['setting_name'] + time.strftime("-%Y-%m-%d_%H_%M_%S")
print('seed = %d' % args.seed)
clf_args['freeze'] = args.freeze

if args.mode=='test': percentages=[0.01,0.05]
elif args.mode=='0.05':percentages=[0.05]
elif args.mode=='all' : percentages=[0.001,0.005,0.01,0.05,0.1,0.2,0.5,1.000]
elif args.mode=='oldtime' : percentages=[0.001,0.005,0.01,0.05,1.0]
elif args.mode=='hundred': percentages=[1]
else: percentages=[0.1,0.2,0.5]
auroc_test = pd.DataFrame(index=[args.seed], columns=percentages)

ap_test = pd.DataFrame(index=[args.seed], columns=percentages)

if clf_args.get('mbest') is None:
    clf_args['mbest'] = cpc_args['model_best']

# percentages=[0.01,0.05]
if args.mode=='all':
    save_name = os.path.join(top_path, 'logs', 'imp', 'FineTune', model_type,
                             clf_args['setting_name'] + 'freeze_%s_seed_%s_mseed_%s_all_mbest_%s.csv' % (
                             args.freeze, args.seed, clf_args['model_seed'],clf_args['mbest']))
elif args.mode=='0.05':
    save_name = os.path.join(top_path, 'logs', 'imp', 'FineTune', model_type,
                             clf_args['setting_name'] + 'freeze_%s_seed_%s_mseed_%s_only0.05_mbest_%s.csv' % (
                                 args.freeze, args.seed, clf_args['model_seed'], clf_args['mbest']))
elif args.mode=='oldtime':
    save_name = os.path.join(top_path, 'logs', 'imp', 'FineTune', model_type,
                         clf_args['setting_name'] + 'freeze_%s_seed_%s_mseed_%s_0.001_0.005_0.1_0.5_1.0_mbest_%s.csv' % (args.freeze, args.seed,clf_args['model_seed'],clf_args['mbest']))
elif args.mode=='test':
    save_name = os.path.join(top_path, 'logs', 'imp', 'FineTune', model_type,
                         clf_args['setting_name'] + 'freeze_%s_seed_%s_mseed_%s_0.01_0.05_mbest_%s.csv' % (args.freeze, args.seed,clf_args['model_seed'],clf_args['mbest']))

else:
    save_name = os.path.join(top_path, 'logs', 'imp', 'FineTune', model_type,
                         clf_args['setting_name'] + 'freeze_%s_seed_%s_mseed_%s_0.1_0.2_0.5_mbest_%s.csv' % (args.freeze, args.seed,clf_args['model_seed'],clf_args['mbest']))
# assert os.path.exists(save_name)==False, "This seed %s is already done successfully!!"%args.seed

for percentage in percentages:
    clf_args['percentage']=percentage
    clf_args['seed']=args.seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if not pd.isna(auroc_test.loc[args.seed, percentage]):
        # continue
    print('-----------------------------\n percentage = %f\n---------------------------' % percentage)
    auc = 0
    train_set, validation_set, test_set = split_Structure_Inhospital(
        cpc_args, percentage, args.seed, out='set'
    )
    while auc == 0:
        auc,p = fine_tune_imp(model,train_set,validation_set,test_set, clf_args, cpc_args, freeze=args.freeze)
        # clf_args['model_seed']+=1
        # auc,p=1,1
    auroc_test.loc[args.seed, percentage] = auc
    ap_test.loc[args.seed, percentage] = p

    print('AUROC')
    print(auroc_test)
    print('Average Precision')
    print(ap_test)
    print('all')
    all=pd.concat([auroc_test,ap_test],axis=1)
    all.to_csv(save_name)
    print(all)
    print('data saved to %s' %save_name)

# print('final model_seed is %s'%clf_args['model_seed'])

