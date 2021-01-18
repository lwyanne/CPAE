from __future__ import print_function
import os, sys
# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# import GPUtil
import fastai.metrics as fastaimetrics
from fastai.callbacks import *
import fastai
from sklearn.linear_model import LogisticRegression

from fastai.basic_train import Learner as fastaiLearner
import numpy as np
from torch.nn.utils.rnn import pack_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from models.utils import *
from models.datareader import *
from models.networks import *
from models.benchmark import *
import inspect
import time
from models.networkSwitch import CPAE1_S,CPAE4_S,CPLSTM3,CPLSTM4,CPAELSTM44,CPAELSTM44_AT,CPAELSTM44_selfAT,CAE_LSTM,AE_LSTM

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score as skkappa
from sklearn.metrics import average_precision_score as ap
from sklearn.metrics import precision_recall_curve
kappa=fastaimetrics.KappaScore()
kappa.weights = "quadratic"



def try_resume_los(args,myLearner,loadLearner):
    if args['resume']:
        try:
            myLearner.load(loadLearner)
        except RuntimeError:
            return 0,0,0,0
        print('Load learner from %s'%loadLearner)
        y = myLearner.get_preds(ds_type=DatasetType.Test)
        preds, labels = y[0].cpu(), y[1].cpu()
        auroc_test = roc_auc_score(labels, preds, multi_class='ovo')
        acc_test = fastaimetrics.accuracy(preds, labels).item()
        kappa_linear_test = skkappa(torch.argmax(preds, dim=1), labels, weights='linear')
        kappa_quadratic_test=skkappa(torch.argmax(preds, dim=1), labels, weights='quadratic')
        return auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test
    else:
        return 0,0,0,0

def try_resume_imp(args,myLearner,loadLearner):
    if args['resume']:
        try:
            myLearner.load(loadLearner)
        except RuntimeError:
            return 0,0
        print('Load learner from %s'%loadLearner)
        y = myLearner.get_preds(ds_type=DatasetType.Test)
        preds, labels = y[0].cpu(), y[1].cpu()
        auroc_test = roc_auc_score(labels, preds[:,1])
        # auprc_test= precision_recall_curve(labels,preds[:,1])
        p_test = ap(labels, preds[:, 1])

        return auroc_test,p_test
    else:
        return 0,0



def get_clf(clf_args,cpc_args,out=2):
    if 'CP' in clf_args['model_setting'] or 'CD' in clf_args['model_setting'] or 'AE_LSTM' in clf_args['model_setting']:
        indim=cpc_args['gru_out']
        if clf_args.get('stack') :
            if cpc_args.get('noct'):
                indim=cpc_args['gru_out']*192
            else:
                indim=cpc_args['gru_out']*193
    else:
        indim=cpc_args['n_flatten_features']



    if clf_args.get("mlp_layers") is None:
        clf=LR(seed=clf_args['model_seed'], in_features=indim,out=out).to(device)
    else:
        drop= clf_args.get("drop")
        clf= MLP(seed=clf_args['model_seed'], hidden_sizes=clf_args['mlp_layers'], in_features=indim,out=out,dropout=drop).to(device)
    return clf



#          -------------------------------------------------------------------------------------------------
#          ------------------------------In-hospital-mortality-prediction-----------------------------------
#          -------------------------------------------------------------------------------------------------
#          ------------------Use the embedded vector learned from CPC to do the prediction

#          ---------------------------------------------------
#          Directly run predict_imp.py to test this------------
#          ---------------------------------------------------




def fine_tune_imp(Model, train_set,val_set,test_set,args, cpc_args,freeze=False):
    #
    #                              -------- set model seed --------
    set_seed(args['model_seed'])
    print('model_seed=%s'%args['model_seed'])

    #                              ---------  check point  --------

    if args.get('update_mseed') is None:
        args['update_mseed'] = 0.5
    lrs,ifbn=get_lr_ifbn(args,cpc_args)
    bs=get_bs_imp(args,len(train_set))
    bunch = fastai_dl(train_set, val_set, test_set, device, batch_size=bs, num_workers=20)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    clf=get_clf(args,cpc_args,out=2) # define classifier
    if args.get('two_step'):
        print('pretrain classifier...')
        train_data,val_data=get_out(Model,train_set,val_set,args,cpc_args,freeze=True,get_test=False)
        lr_clf = LogisticRegression(random_state=0, n_jobs=30, max_iter=100).fit(train_data[0], train_data[1])
        print(lr_clf.coef_)
        weights=torch.cat((-torch.tensor(lr_clf.coef_).to(device),torch.tensor(lr_clf.coef_).to(device)),0)
        bias=torch.tensor([1,0])
        items = [(k, v.shape) for k, v in clf.state_dict().items()][:]
        new_state_dict = {items[0][0]: weights,items[1][0]:bias}
        clf.load_state_dict(new_state_dict, strict=False)
        # print(clf.state_dict().items()[0])
        print('load classier done...')
    if args.get('pre_train'):
        checkpoint_path = os.path.join(
            cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
            args['mbest'])
        checkpoint = torch.load(checkpoint_path)
        freeze_partial = load_model_imp(cpc, checkpoint['state_dict'], args)

        print('model will be loaded from %s'%checkpoint_path)
    else:
        freeze_partial=None
    model = CPclassifier(cpc, clf,bn=ifbn,freeze=freeze,stack=args.get('stack'),warm=args.get('warm'),conti=args.get('conti'),partial=freeze_partial, ifrelu=args.get('ifrelu'))
    print(str(model))
    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy],
                              callback_fns=[biAUROC, partial(EarlyStoppingCallback, monitor='AUROC', min_delta=0.002,
                                                             patience=15)])
    # discriminate_leaner(myLearner,[myLearner.model.CPmodel.lstm1,myLearner.model.CPmodel.lstm2,myLearner.model.MLP],cpc_args['model_type'])
    # myLearner = fastaiLearner(bunch, model=model,
    #                           loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy],
    #                           callback_fns=[biAUROC])


    # myLearner = fastaiLearner(bunch, model=model,
    #                           loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy],
    #                           callback_fns=[biAUROC, partial(EarlyStoppingCallback, monitor='AUROC', min_delta=0.002,
    #                                                          patience=10)])
    bestLearner,tempLearner=define_save_path_imp(args,lrs)


    # # Test needed
    #
    #
    #  # lrs=args['lrs']
    # if args['resume']:
    #     auroc_test, ap_test = try_resume_imp(args, myLearner, bestLearner)
    #     if auroc_test and args['percentage']<1:
    #         print('Load learner successfully')
    #         return auroc_test, ap_test
    #     elif auroc_test and int(args['percentage'])==1:
    #         print('load learner successfully for 100% training data, continue training')
    # print('Load learner failed....\nStart to re-train...')


    if args.get('update_mseed') != False:
        if args.get('update_mseed') is None:
            args['update_mseed']=0.5
        print("Check if this trial is okay at threshold %s......."%args['update_mseed'])
        myLearner.fit(1, lrs, callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='AUROC',
                                                           name=tempLearner)])
        y = myLearner.get_preds(ds_type=DatasetType.Valid)
        preds, labels = y[0], y[1]
        s_val_temp = roc_auc_score(labels, preds[:, 1])
        y = myLearner.get_preds(ds_type=DatasetType.Test)
        preds, labels = y[0], y[1]
        s_test_temp = roc_auc_score(labels, preds[:, 1])
        p_test_temp = ap(labels, preds[:, 1])
        if s_val_temp<args['update_mseed']:
            print('This trial seems to fail...')
            args['model_seed']+=1
            # args['seed']+=1
            print('model_seed+=1 and start to retrain')
            return 0,0
        print('This trial is okay, mseed will not be udpated....\nss')
    else:
        s_test_temp=-np.inf
        p_test_temp=-np.inf


    #
    if len(train_set)<20 and args.get('two_step'):
        args['epochs']=10
        print('epochs is set as 10')
    myLearner.fit_one_cycle(args['epochs'], max_lr=lrs,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='AUROC',
                                                         name=bestLearner)])
    # myLearner.fit(args['epochs'], lr=lrs,
    #                         callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='AUROC',
    #                                                      name=bestLearner)])


    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Valid)
    preds, labels = y[0], y[1]
    s_val = roc_auc_score(labels,preds[:,1])
    p_val= ap(labels,preds[:,1])
    del y
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds, labels = y[0], y[1]
    s_test = roc_auc_score(labels,preds[:,1])
    p_test= ap(labels,preds[:,1])


    if s_test>s_test_temp :
        print('for ini %s\n,The auroc for validation is %s, the auroc for test is %s' % (args['setting_name'], s_val, s_test))
        return  s_test,p_test
    else:
        print('the best model is the first attempt')
        print('for ini %s\n,The auroc for validation is %s, the auroc for test is %s' % (args['setting_name'], s_val_temp, s_test_temp))
        return s_test_temp,p_test_temp


def fine_tune_los(Model,train_set,validation_set,test_set, args, percentage, cpc_args,num_per_class=None,total=None,freeze=False,seed=None):
    torch.manual_seed(args['model_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args['len_train']=len(train_set)
    if args.get('mbest') is None:
        args['mbest'] = cpc_args['model_best']
    if args['pre_train']:
        checkpoint_path = os.path.join(cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
                                       args['mbest'])
        print('loading model from checkpoint %s'%checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
    freeze_partial=None
    logging_dir = os.path.join(cpc_args['top_path'], 'logs', 'los', 'FineTune', cpc_args['model_type'])
    run_name = args['run_name']
    setting_name = args['setting_name']
    clf=get_clf(args,cpc_args,out=10)
    epochs = args['epochs']
    try:lrs = args['lrs']
    except KeyError: lrs=args['lr']
    print('learning rates are %s' %str(lrs))
    print('model_seed=%s'%args['model_seed'])
    bs=args['bs']

    # Sample the training set with pre-defined percentage
    if len(train_set)<100:
        bs=10
        n_work=3
    else:
        bs=50
        n_work=3
    bunch = fastai_dl(train_set, validation_set, test_set, device, batch_size=bs, num_workers=n_work)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    if args['pre_train']:
        if args.get('load_reg') != False:
            cpc.load_state_dict(checkpoint['state_dict'])
            print('\n-----------\nload parameters of pretrained model.lstm1 & lstm2....\n')
        else:
            load_only_encode(cpc,checkpoint['state_dict'])
            print('\n-----------\nOnly load parameters of pretrained model.lstm1.....\n')# load
            if args['freeze']: freeze_partial='encode'
            print('model.lstm1 is freeze')

    bestLearner, _ = define_save_path_los(args, lrs)

    model = CPclassifier(cpc, clf,bn=args.get('ifbn'),freeze=freeze,stack=args.get('stack'),warm=args.get('warm'),conti=args.get('conti'),partial=freeze_partial,ifrelu=args.get('ifrelu'))

    print(str(model))

    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy,kappa],
                              callback_fns=[AUROC,partial(EarlyStoppingCallback, monitor='kappa_score', min_delta=0.002, patience=8)])

    if 'LSTM' in cpc_args['model_type']:
        l=[myLearner.model.CPmodel.lstm1,myLearner.model.CPmodel.lstm2,myLearner.model.MLP]
        myLearner.split(split_on=l)
    if total is not None or num_per_class is not None:
        # "ALL needs rerun"
        auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test= try_resume_los(args, myLearner, bestLearner)
        if auroc_test:
            print('Load learner successfully')
            return auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test

    print('Load learner failed....\nStart to re-train...')
    bestLearner = os.path.join(logging_dir, '%s_%s_%s_freeze_%s-best' % (args['setting_name'],len(train_set),seed,args['freeze']))
    print('The learner will be saved as %s'%bestLearner)

    if args.get('update_mseed') != False:
        if args.get('update_mseed') is None:
            args['update_mseed'] = 0.5
        print("Check if this trial is okay at threshold %s......." % args['update_mseed'])
        myLearner.fit(1,0.001)
        y = myLearner.get_preds(ds_type=DatasetType.Valid)
        preds,labels=y[0],y[1]
        auroc_val = roc_auc_score(labels,preds,multi_class='ovo')
        if auroc_val<args['update_mseed']:
            print('This trial failed!')
            args['model_seed']+=1
            return 0,0,0,0
        print("This trial is okay, mseed will not be updated....\n")

    myLearner.fit_one_cycle(epochs, lrs,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='kappa_score',
                                                         name=bestLearner)])
    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds,labels=y[0].cpu(),y[1].cpu()
    auroc_test = roc_auc_score(labels, preds, multi_class='ovo')
    acc_test = fastaimetrics.accuracy(preds, labels).item()
    kappa_linear_test = skkappa(torch.argmax(preds, dim=1), labels, weights='linear')
    kappa_quadratic_test = skkappa(torch.argmax(preds, dim=1), labels, weights='quadratic')

    print('kappa_linear,kappa_quadratic,auroc,acc\n')
    print(kappa_quadratic_test,kappa_linear_test,auroc_test,acc_test)
    return auroc_test, acc_test, kappa_linear_test, kappa_quadratic_test

def fine_tune_los_bi(Model,train_set,validation_set,test_set, args, cpc_args,freeze=False,seed=None):
    torch.manual_seed(args['model_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args['len_train']=len(train_set)
    if args.get('mbest') is None:
        args['mbest'] = cpc_args['model_best']
    checkpoint_path = os.path.join(cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
                                   args['mbest'])
    print('loading model from checkpoint %s'%checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    freeze_partial=None
    logging_dir = os.path.join(cpc_args['top_path'], 'logs', 'los', 'FineTune', cpc_args['model_type'])
    run_name = args['run_name']
    setting_name = args['setting_name']
    clf=get_clf(args,cpc_args,out=2)
    epochs = args['epochs']
    try:lrs = args['lrs']
    except KeyError: lrs=args['lr']
    print('learning rates are %s' %str(lrs))
    print('model_seed=%s'%args['model_seed'])
    bs=args['bs']

    # Sample the training set with pre-defined percentage
    if len(train_set)<100:
        bs=10
        n_work=3
    else:
        bs=50
        n_work=3
    bunch = fastai_dl(train_set, validation_set, test_set, device, batch_size=bs, num_workers=n_work)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    if args['pre_train']:
        if args.get('load_reg') != False:
            cpc.load_state_dict(checkpoint['state_dict'])
            print('\n-----------\nload parameters of pretrained model.lstm1 & lstm2....\n')
        else:
            load_only_encode(cpc,checkpoint['state_dict'])
            print('\n-----------\nOnly load parameters of pretrained model.lstm1.....\n')# load
            if args['freeze']: freeze_partial='encode'
            print('model.lstm1 is freeze')

    bestLearner, _ = define_save_path_los(args, lrs)

    model = CPclassifier(cpc, clf,bn=args.get('ifbn'),freeze=freeze,stack=args.get('stack'),warm=args.get('warm'),conti=args.get('conti'),partial=freeze_partial)

    print(str(model))

    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy,kappa],
                              callback_fns=[biAUROC,partial(EarlyStoppingCallback, monitor='AUROC', min_delta=0.002, patience=5)])

    if 'LSTM' in cpc_args['model_type']:
        l=[myLearner.model.CPmodel.lstm1,myLearner.model.CPmodel.lstm2,myLearner.model.MLP]
        myLearner.split(split_on=l)


  # try resume
    auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test= try_resume_los(args, myLearner, bestLearner)
    if auroc_test:
        print('Load learner successfully')
        return auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test

    print('Load learner failed....\nStart to re-train...')
    bestLearner = os.path.join(logging_dir, '%s_%s_%s_freeze_%s-best' % (args['setting_name'],len(train_set),seed,args['freeze']))
    print('The learner will be saved as %s'%bestLearner)

    if args.get('update_mseed') != False:
        if args.get('update_mseed') is None:
            args['update_mseed'] = 0.5
        print("Check if this trial is okay at threshold %s......." % args['update_mseed'])
        myLearner.fit(1,0.001)
        y = myLearner.get_preds(ds_type=DatasetType.Valid)
        preds,labels=y[0],y[1]
        auroc_val = roc_auc_score(labels, preds[:, 1])
        if auroc_val<args['update_mseed']:
            print('This trial failed!')
            args['model_seed']+=1
            return 0,0,0,0
        print("This trial is okay, mseed will not be updated....\n")

    myLearner.fit_one_cycle(epochs, lrs,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='AUROC',
                                                         name=bestLearner)])
    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds,labels=y[0].cpu(),y[1].cpu()
    auroc_test =roc_auc_score(labels, preds[:, 1])
    acc_test = fastaimetrics.accuracy(preds, labels).item()
    kappa_linear_test = skkappa(torch.argmax(preds, dim=1), labels, weights='linear')
    kappa_quadratic_test = skkappa(torch.argmax(preds, dim=1), labels, weights='quadratic')

    print('kappa_linear,kappa_quadratic,auroc,acc\n')
    print(kappa_quadratic_test,kappa_linear_test,auroc_test,acc_test)
    return auroc_test, acc_test, kappa_linear_test, kappa_quadratic_test


def fine_tune_los_tri(Model,train_set,validation_set,test_set, args, cpc_args,freeze=False,seed=None):
    torch.manual_seed(args['model_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args['len_train']=len(train_set)
    if args.get('mbest') is None:
        args['mbest']=cpc_args['model_best']

    freeze_partial=None
    logging_dir = os.path.join(cpc_args['top_path'], 'logs', 'los', 'FineTune', cpc_args['model_type'])
    run_name = args['run_name']
    setting_name = args['setting_name']
    clf=get_clf(args,cpc_args,out=3)
    epochs = args['epochs']
    try:lrs = args['lrs']
    except KeyError: lrs=args['lr']
    print('learning rates are %s' %str(lrs))
    print('model_seed=%s'%args['model_seed'])
    bs=args['bs']
    if float(args.get('percentage'))==0.01:patience = 10
    else: patience=10
    # Sample the training set with pre-defined percentage
    if len(train_set)<100:
        bs=10
        n_work=3
    else:
        bs=50
        n_work=3
    bunch = fastai_dl(train_set, validation_set, test_set, device, batch_size=bs, num_workers=n_work)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    if args['pre_train']:
        checkpoint_path = os.path.join(cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
                                       args['mbest'])
        print('loading model from checkpoint %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        if args.get('load_reg') != False:
            cpc.load_state_dict(checkpoint['state_dict'])
            print('\n-----------\nload parameters of pretrained model.lstm1 & lstm2....\n')
        else:
            load_only_encode(cpc,checkpoint['state_dict'])
            print('\n-----------\nOnly load parameters of pretrained model.lstm1.....\n')# load
            if args['freeze']: freeze_partial='encode'
            print('model.lstm1 is freeze')

    bestLearner, _ = define_save_path_los(args, lrs)

    model = CPclassifier(cpc, clf,bn=args.get('ifbn'),freeze=freeze,stack=args.get('stack'),warm=args.get('warm'),conti=args.get('conti'),partial=freeze_partial,ifrelu=args.get('ifrelu'))
    print('patience is set to %s'%patience)
    print(str(model))

    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy,kappa],
                              callback_fns=[AUROC,partial(EarlyStoppingCallback, monitor='kappa_score', min_delta=0.002, patience=15)])
    # myLearner = fastaiLearner(bunch, model=model,
    #                           loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy,kappa],
    #                           callback_fns=[AUROC])
    # patience was 5. but now we set to 8 for the bad seed.

    if 'LSTM' in cpc_args['model_type']:
        l=[myLearner.model.CPmodel.lstm1,myLearner.model.CPmodel.lstm2,myLearner.model.MLP]
        myLearner.split(split_on=l)


  # try resume
    auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test= try_resume_los(args, myLearner, bestLearner)
    if auroc_test:
        print('Load learner successfully')
        return auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test

    print('Load learner failed....\nStart to re-train...')
    bestLearner = os.path.join(logging_dir, '%s_%s_%s_freeze_%s-best' % (args['setting_name'],len(train_set),seed,args['freeze']))
    print('The learner will be saved as %s'%bestLearner)

    if args.get('update_mseed') != False:
        if args.get('update_mseed') is None:
            args['update_mseed'] = 0.5
        print("Check if this trial is okay at threshold %s......." % args['update_mseed'])
        myLearner.fit(1,0.001)
        y = myLearner.get_preds(ds_type=DatasetType.Valid)
        preds,labels=y[0],y[1]
        auroc_val = roc_auc_score(labels,preds,multi_class='ovo')
        if auroc_val<args['update_mseed']:
            print('This trial failed!')
            args['model_seed']+=1
            return 0,0,0,0
        print("This trial is okay, mseed will not be updated....\n")

    myLearner.fit_one_cycle(60, lrs,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='kappa_score',
                                                         name=bestLearner)])
    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds,labels=y[0].cpu(),y[1].cpu()
    auroc_test = roc_auc_score(labels, preds, multi_class='ovo')
    acc_test = fastaimetrics.accuracy(preds, labels).item()
    kappa_linear_test = skkappa(torch.argmax(preds, dim=1), labels, weights='linear')
    kappa_quadratic_test = skkappa(torch.argmax(preds, dim=1), labels, weights='quadratic')

    print('kappa_linear,kappa_quadratic,auroc,acc\n')
    print(kappa_quadratic_test,kappa_linear_test,auroc_test,acc_test)
    return auroc_test, acc_test, kappa_linear_test, kappa_quadratic_test


def fine_tune_los_reg_all(Model,train_set,validation_set,test_set, args, cpc_args,freeze=False,seed=None):
    #                              -------- set model seed --------
    set_seed(args['model_seed'])
    print('model_seed=%s' % args['model_seed'])

    logging_dir = os.path.join(cpc_args['top_path'], 'logs', 'los', 'FineTune', cpc_args['model_type'])
    if args.get('mbest') is None:
        args['mbest'] = cpc_args['model_best']
    checkpoint_path = os.path.join(cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
                                   args['mbest'])
    print('loading model from checkpoint %s'%checkpoint_path)
    checkpoint = torch.load(checkpoint_path)


    if args.get('update_mseed') is None:
        args['update_mseed'] = 0.5



    freeze_partial=None
    run_name = args['run_name']
    setting_name = args['setting_name']
    clf=get_clf(args,cpc_args,out=1)
    epochs = args['epochs']
    try:lrs = args['lrs']
    except KeyError: lrs=args['lr']
    print('learning rates are %s' %str(lrs))
    print('model_seed=%s'%args['model_seed'])
    bs=args['bs']

    # Sample the training set with pre-defined percentage
    if len(train_set)<100:
        bs=10
        n_work=3
    else:
        bs=50
        n_work=3
    bunch = fastai_dl(train_set, validation_set, test_set, device, batch_size=bs, num_workers=n_work)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    if args['pre_train']:
        if args.get('load_reg') != False:
            cpc.load_state_dict(checkpoint['state_dict'])
            print('\n-----------\nload parameters of pretrained model.lstm1 & lstm2....\n')
        else:
            load_only_encode(cpc,checkpoint['state_dict'])
            print('\n-----------\nOnly load parameters of pretrained model.lstm1.....\n')# load
            if args['freeze']: freeze_partial='encode'
            print('model.lstm1 is freeze')
    args['len_train']=len(train_set)
    bestLearner, _ = define_save_path_los(args, lrs)

    model = CPclassifier(cpc, clf,bn=args.get('ifbn'),freeze=freeze,stack=args.get('stack'),warm=args.get('warm'),conti=args.get('conti'),partial=freeze_partial)

    print(str(model))

    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.MSELoss(), metrics=[fastaimetrics.mse,fastaimetrics.mae],
                              callback_fns=[partial(EarlyStoppingCallback, monitor='mean_absolute_error', min_delta=0.002, patience=8)])

    if 'LSTM' in cpc_args['model_type']:
        l=[myLearner.model.CPmodel.lstm1,myLearner.model.CPmodel.lstm2,myLearner.model.MLP]
        myLearner.split(split_on=l)
    # if total is not None or num_per_class is not None:
    #     # "ALL needs rerun"
    #     auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test= try_resume_los(args, myLearner, bestLearner)
    #     if auroc_test:
    #         print('Load learner successfully')
    #         return auroc_test,acc_test,kappa_linear_test,kappa_quadratic_test

    # print('Load learner failed....\nStart to re-train...')
    bestLearner = os.path.join(logging_dir, 'reg_all_%s_%s_%s_freeze_%s-best' % (args['setting_name'],len(train_set),seed,args['freeze']))
    print('The learner will be saved as %s'%bestLearner)



    myLearner.fit_one_cycle(epochs, lrs,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='mean_absolute_error',
                                                         name=bestLearner)])
    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds,labels=y[0].cpu(),y[1].cpu()

    mse_test=fastaimetrics.mse(preds,labels)
    mae_test=fastaimetrics.mae(preds,labels)

    print('mse , mae\n')
    print(mse_test,mae_test)
    return mse_test,mae_test




def fine_tune_dd(Model, args, percentage, cpc_args,freeze=False,seed=None):
    checkpoint_path = os.path.join(
        cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
        cpc_args['model_best'])
    checkpoint = torch.load(checkpoint_path)

    logging_dir = os.path.join(cpc_args['top_path'], 'logs', 'dd', 'FineTune', cpc_args['model_type'])
    run_name = args['run_name']
    setting_name = args['setting_name']
    mlp = MLP(args['mlp_layers'], in_features=cpc_args['gru_out']).to(device)
    epochs = args['epochs']
    lr = args['lr']

    # Sample the training set with pre-defined percentage
    train_set, validation_set, test_set = split_Structure_los(
        cpc_args, percentage, random_seed=seed, out='set'
    )
    print('The samples size of training set == %d' % len(train_set))
    if len(train_set) < 100:
        bs = 10
        n_work = 30
    else:
        bs = 50
        n_work = 30
    bunch = fastai_dl(train_set, validation_set, test_set, device, batch_size=bs, num_workers=n_work)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    if args['pre_train']: cpc.load_state_dict(checkpoint['state_dict'])  # load

    model = CPclassifier(cpc, mlp, freeze)
    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy, kappa],
                              callback_fns=AUROC)
    bestLearner = os.path.join(logging_dir, '%s_%s_freeze_%s-best' % (len(train_set), seed, args['freeze']))
    print('The learner is saved as %s' % bestLearner)
    myLearner.fit(1, 0.001)
    y = myLearner.get_preds(ds_type=DatasetType.Valid)
    preds, labels = y[0], y[1]
    auroc_val = roc_auc_score(labels, preds, multi_class='ovo')
    if auroc_val < 0.5:
        print('This trial failed!')
        return 0, 0, 0
    myLearner.fit_one_cycle(epochs, lr,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='kappa_score',
                                                         name=bestLearner)])
    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds, labels = y[0].cpu(), y[1].cpu()
    auroc_test = roc_auc_score(labels, preds, multi_class='ovo')
    acc_test = fastaimetrics.accuracy(preds, labels).item()
    kappa_test = skkappa(torch.argmax(preds, dim=1), labels)

    y = myLearner.get_preds(ds_type=DatasetType.Valid)
    preds, labels = y[0].cpu(), y[1].cpu()
    auroc_val = roc_auc_score(labels, preds, multi_class='ovo')
    acc_val = fastaimetrics.accuracy(preds, labels).item()
    kappa_val = skkappa(torch.argmax(preds, dim=1), labels)

    print('for ini %s\n,The auroc for validation is %s, the auroc for test is %s' % (setting_name, auroc_val, auroc_test))
    print('for ini %s\n,The accuracy for validation is %s, the accuracy for test is %s' % (setting_name, acc_val, acc_test))
    print('for ini %s\n,The kappa for validation is %s, the kappa for test is %s' % (setting_name, kappa_val, kappa_test))

    return auroc_test, acc_test, kappa_test


def end2end_imp(Model, train_set,val_set,test_set, percentage, cpc_args,freeze=False,seed=None):
    #                                  -------- set model seed --------
    torch.manual_seed(cpc_args['model_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cpc_args['model_seed'])
    #                              ---------  check point  --------
    if cpc_args['model_best'] is not None:
        checkpoint_path = os.path.join(
            cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
            cpc_args['model_best'])
        checkpoint = torch.load(checkpoint_path)

    logging_dir = os.path.join(cpc_args['top_path'], 'logs', 'imp', 'FineTune', cpc_args['model_type'])
    setting_name = cpc_args['setting_name']
    # clf=get_clf(args,cpc_args,out=2)
    epochs = cpc_args['epochs']
    lr = cpc_args['lr']
    print('model_seed=%s'%cpc_args['model_seed'])
    print('lr==%s'%str(lr))
    if int(percentage)==1: epochs=80
    # Sample the training set with pre-defined percentage

    bunch = fastai_dl(train_set, val_set, test_set, device, batch_size=12, num_workers=30)    # fastai dataloader
    model, optimizer = define_model(cpc_args, Model, train_set)     # define model using params in cpc_args
    # if args['pre_train']: cpc.load_state_dict(checkpoint['state_dict'])  # load

    # model = CPclassifier(cpc, clf,freeze)
    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy],
                              callback_fns=[biAUROC,partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.003, patience=5)])
    # l=[myLearner.model.CPmodel.lstm1,myLearner.model.CPmodel.lstm2,myLearner.model.MLP]
    # myLearner.split(split_on=l)
    bestLearner = os.path.join(logging_dir, '%s_%s_seed_%s_modelseed_%s_freeze_%s-best' % (setting_name,percentage,seed,cpc_args['model_seed'],cpc_args['freeze']))
    tempLearner = os.path.join(logging_dir, '%s_%s_seed_%s_modelseed_%s_freeze_%s-temp' % (setting_name,percentage,seed,cpc_args['model_seed'],cpc_args['freeze']))
    lr=cpc_args['lr']
    if cpc_args['resume']:
        auroc_test, ap_test = try_resume_imp(cpc_args, myLearner, bestLearner)
        if auroc_test and percentage<1:
            print('Load learner successfully')
            return auroc_test, ap_test
        elif auroc_test and int(percentage)==1:
            print('load learner successfully for 100% training data, continue training')
    print('Set epochs to 80 for 100% training set ')
    print('Load learner failed....\nStart to re-train...')

    myLearner.fit(1,0.001,callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='AUROC',
                                                         name=tempLearner)])
    y = myLearner.get_preds(ds_type=DatasetType.Valid)
    preds, labels = y[0], y[1]
    s_val_temp = roc_auc_score(labels,preds[:,1])
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds, labels = y[0], y[1]
    s_test_temp = roc_auc_score(labels, preds[:, 1])
    p_test_temp= ap(labels,preds[:,1])
    auprc_test_temp = precision_recall_curve(labels, preds[:, 1])

    if s_test_temp<0.5:
        print('This trial seems to fail...')
    myLearner.fit_one_cycle(epochs, max_lr=lr,
                            callbacks=[SaveModelCallback(myLearner, every='improvement', monitor='AUROC',
                                                         name=bestLearner)])
    myLearner.load(bestLearner)
    y = myLearner.get_preds(ds_type=DatasetType.Valid)
    preds, labels = y[0], y[1]
    s_val = roc_auc_score(labels,preds[:,1])
    p_val= ap(labels,preds[:,1])
    del y
    y = myLearner.get_preds(ds_type=DatasetType.Test)
    preds, labels = y[0], y[1]
    s_test = roc_auc_score(labels,preds[:,1])
    p_test= ap(labels,preds[:,1])

    if s_test>s_test_temp:
        print('for ini %s\n,The auroc for validation is %s, the auroc for test is %s' % (setting_name, s_val, s_test))

        return  s_test,p_test
    else:
        print('the best model is the first attempt')
        print('for ini %s\n,The auroc for validation is %s, the auroc for test is %s' % (setting_name, s_val_temp, s_test_temp))
        return s_test_temp,p_test_temp



def get_out(Model,train_set,test_set, args, cpc_args,freeze=False,get_test=False):
    set_seed(1)
    logger=logging.getLogger('lr')
    logger.info('model_seed=1')


    lrs, ifbn = 1,False
    bs = get_bs_imp(args, len(train_set))
    bunch = fastai_dl(train_set, test_set, test_set, device, batch_size=bs, num_workers=20)
    cpc, optimizer = define_model(cpc_args, Model, train_set)
    if args.get('pre_train'):
        checkpoint_path = os.path.join(
            cpc_args['top_path'], 'logs', 'cpc', cpc_args['model_type'],
            args['mbest'])
        checkpoint = torch.load(checkpoint_path)
        freeze_partial = load_model_imp(cpc, checkpoint['state_dict'], args)

        logger.info('model will be loaded from %s' % checkpoint_path)
    else:
        freeze_partial = None
    model = CPout(cpc, bn=ifbn, freeze=freeze, stack=args.get('stack'), warm=args.get('warm'),
                         conti=args.get('conti'), partial=freeze_partial, ifrelu=args.get('ifrelu'))
    logger.info(str(model))
    myLearner = fastaiLearner(bunch, model=model,
                              loss_func=nn.CrossEntropyLoss(), metrics=[fastaimetrics.accuracy],
                              callback_fns=[biAUROC])

    train_data = myLearner.get_preds(ds_type=DatasetType.Train)
    logger.info('train set shape =%s'%str(train_data[0].shape))
    if get_test:
        test_data= myLearner.get_preds(ds_type=DatasetType.Valid)
        logger.info('test set shape =%s'%str(test_data[0].shape))
        r=(train_data,test_data)
    else:
        r=(train_data,None)



    return r
