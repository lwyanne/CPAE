import os, sys

# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import configparser
from models.utils import *
from models.downstream import *
from models.networks import *
from models.networkSwitch import CPAELSTM44
from models.benchmark import *
from joblib import Parallel, delayed


torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--ini-file', default=None, required=True,
                    help='the path to the setting9.ini')
parser.add_argument('--resume', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--load-intermediate-data', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--freeze', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--model-seed', default=0, type=int)
parser.add_argument('--mode', default=None, required=False,
                    help='whether to just test percentage at 0.05,0.001')
args = parser.parse_args()

print('===========================================')
print('==========LOS  tri outcome=============')
print('Fine-tune according to %s'%args.ini_file)
print('Freeze = %s '%args.freeze)
if args.mode=='test': percentages=[0.05]
elif args.mode=='all' : percentages=[0.005,0.01,0.05,0.1,0.2,0.5,1.000]
elif args.mode=='oldtime' : percentages=[0.005,0.01,0.05,1.0]
elif args.mode=='hundred': percentages=[1]
elif args.mode=='bad': percentages=[0.01]
else: percentages=[0.1,0.2,0.5]

# parser.add_argument('--id', default=False, type=int)

# parser.add_argument('--percentage', default=1)
# parser.add_argument('--sampletimes', default=1)

clf_args = get_config_dic(args.ini_file)
cpc_args_file = os.path.join(top_path,clf_args['model_setting'])
model_type = get_model_type(cpc_args_file)
model = eval(model_type)
cpc_args = get_config_dic(cpc_args_file)


if clf_args.get('model_seed') is None:
    clf_args['model_seed']=args.model_seed
if model_type == 'CDCK2':
    assert cpc_args['conv_sizes'][-1] % cpc_args['n_frames'] == 0  # THe final output channels should be multiples of the n_frames
cpc_args['learning_rate'] = None
cpc_args['model_type'] = model_type
cpc_args['top_path'] = script_dir
cpc_args['scr_data_path'] = data_dir
clf_args['freeze'] = args.freeze
clf_args['resume']=args.resume
clf_args['setting_name'] = 'EarlyStopping15_'+get_setting_name(os.path.split(args.ini_file)[1])
clf_args['run_name'] = cpc_args['setting_name'] + time.strftime("-%Y-%m-%d_%H_%M_%S")
clf_args['seed'] = args.seed
clf_args['mode']= args.mode
print('run_name is %s'%clf_args['run_name'])
print('seed = %d' % args.seed)

if clf_args.get('mbest') is None:
    clf_args['mbest'] = cpc_args['model_best']


#===============   CHECKING ==================================

# ================
auroc_tests = pd.DataFrame(index=[args.seed],columns=list(map(lambda x:'auroc_'+str(x),percentages)))
acc_tests = pd.DataFrame(index=[args.seed],columns=list(map(lambda x:'acc_'+str(x),percentages)))
kappa_linears = pd.DataFrame(index=[args.seed],columns=list(map(lambda x:'kappa_linear_'+str(x),percentages)))
kappa_quadratics = pd.DataFrame(index=[args.seed],columns=list(map(lambda x:'kappa_qua_'+str(x),percentages)))
prefix=clf_args['setting_name']+'_freeze_%s'%args.freeze+'_percentages_'+str(percentages)+'model_seed_%s'%clf_args['model_seed']+'_seed_%d'%args.seed+'+mbest_'+clf_args['mbest']
suffix='.csv'
save_name=prefix+'tri'+suffix
assert os.path.exists(save_name)==False, "This seed %s is already done successfully!!"%args.seed



for i,percentage in enumerate(percentages):
    clf_args['percentage']=percentage
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_set, validation_set, test_set = split_Structure_los_tri(
        cpc_args, percentage, random_seed=args.seed, out='set'
    )

    print('The samples size of training set == %d' % len(train_set))
    print('\n\n-----------------------------------percentage%s'%percentage)
    auroc_test=0
    while auroc_test==0:
        auroc_test,acc_test,kappa_linear,kappa_quadratic = fine_tune_los_tri(model,train_set,validation_set,test_set, clf_args, cpc_args,freeze=args.freeze,seed=args.seed)
    auroc_tests.iloc[0,i]=auroc_test
    acc_tests.iloc[0,i]=acc_test
    kappa_linears.iloc[0,i]=kappa_linear
    kappa_quadratics.iloc[0,i]=kappa_quadratic
    all = pd.concat([kappa_linears, kappa_quadratics, acc_tests, auroc_tests], axis=1)
    all.to_csv(os.path.join(top_path, 'logs', 'los', 'FineTune', model_type, save_name))

print('----------------------Results--------------AUROC_tests----------------------\n%s'%auroc_tests)
print('----------------------Results--------------ACC_tests----------------------\n%s'%acc_tests)
print('----------------------Results--------------KAPPA_linears----------------------\n%s'%kappa_linears)
print('----------------------Results--------------kappa_quadratics----------------------\n%s'%kappa_quadratics)


print('results saved in %s'%save_name)



