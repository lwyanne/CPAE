import os, sys

# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import time
from models.utils import *
from models.networks import *
from models.networkSwitch import CPLSTM

from models.datareader import *
from models.optimizer import *
logger = logging.getLogger("cpc")

torch.cuda.empty_cache()


def cpc_main(Model, args_json):
    setting_name = args_json['setting_name']
    run_name = args_json['run_name']
    logger = setup_logs(args_json['logging_dir'],'cpc', args_json['run_name'])  # setup logs
    logging_dir = args_json['logging_dir']
    global_timer = timer()  # global timer

    logger.info('===> use %s strategy to split train, validation and eval dataset' % args_json['data_split'])

    train_loader, validation_loader, test_loader = split_Structure_Inhospital(args_json)
    model, optimizer = define_model(args_json, Model,train_loader)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    logger.info(args_json)
    #           ---------------------Load model checkpoint to resume training-----------------

    previous_epoch = 1
    previous_acc = 0
    previous_loss = np.inf
    if args_json['resume']:
        checkpoint_path = os.path.join(
            args_json['logging_dir'],
            args_json['model_best'])
        checkpoint = torch.load(checkpoint_path)
        previous_epoch = checkpoint['epoch']
        previous_loss = checkpoint['validation_loss']
        previous_acc = checkpoint['validation_acc']
        if isinstance(model, CDCK2):
            model.update_flat_features(args_json['n_flat_features_per_frame'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info('\n\nResume from checkpoint: %s\n\n' % checkpoint_path)


    ###          ---------------------------Train and Validate---------------------------------
    best_acc = previous_acc
    best_loss = previous_loss
    best_epoch = previous_epoch
    current_acc=0
    for epoch in range(previous_epoch, args_json['epochs'] + 1):
        epoch_timer = timer()

        train(args_json, model, device, train_loader, optimizer, epoch, args_json['batch_size'],
              args_json['learning_rate'])

        val_acc, val_loss = validation(model, args_json,device, validation_loader)
        logger.info(
            '-------------------------------\nValidation Epoch: {} \tAccuracy: {:.4f}\tLoss: {:.6f}\n------------------'.format(
                epoch, val_acc, val_loss))

        # Saving a General Checkpoint for Inference and/or Resuming Training
        if 'CP' in args_json['model_type'] and val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            snapshot(args_json['logging_dir'], run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            args_json['model_best'] = run_name + '-model_best.pth'  # update json file
            write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
            logger.info("Best model updated in '%s.ini'! !!!!!!!!!" % args_json['setting_name'])
            best_epoch = epoch + 1
        elif 'CP' not in args_json['model_type'] and val_loss< best_loss:
            best_loss=min(val_loss,best_loss)
            snapshot(args_json['logging_dir'], run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            args_json['model_best'] = run_name + '-model_best.pth'  # update json file
            write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
            logger.info("Best model updated in '%s.ini'! !!!!!!!!" % args_json['setting_name'])
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1



        if 'CP' in args_json['model_type'] and val_acc > current_acc:
            current_acc = max(val_acc, current_acc)
            snapshot(args_json['logging_dir'], run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            # args_json['model_best'] = run_name + '-model_best.pth'  # update json file
            # write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
            logger.info("model got improved for this run, but did not surpass previous though, please take changes of hyper parameters into consideration")


        if args_json['save_every'] is not None and epoch%int(args_json['save_every'])==0:
            snapshot(args_json['logging_dir'], run_name+'_epoch_%d'%epoch, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            logger.info('Save this model as %s _epoch_%d-model_best.pth'%(run_name,epoch))
        end_epoch_timer = timer()
        logger.info(
            "#### End epoch {}/{}, elapsed time: {}".format(epoch, args_json['epochs'], end_epoch_timer - epoch_timer))

    ## end 
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

    write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
    logger.info("Parameters updated to '%s.ini'! !!!!!!!!" % args_json['setting_name'])

def cpc_full_main(Model, args_json):
    setting_name = args_json['setting_name']
    run_name = args_json['run_name']
    logger = setup_logs(args_json['logging_dir'],'cpc', args_json['run_name'])  # setup logs
    logging_dir = args_json['logging_dir']
    global_timer = timer()  # global timer

    logger.info('===> use %s strategy to split train, validation and eval dataset' % args_json['data_split'])

    train_loader, validation_loader, test_loader = read_full_seq(args_json)
    model, optimizer = define_model(args_json, Model,train_loader)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    logger.info(args_json)
    #           ---------------------Load model checkpoint to resume training-----------------

    previous_epoch = 1
    previous_acc = 0
    previous_loss = np.inf
    if args_json['resume']:
        checkpoint_path = os.path.join(
            args_json['logging_dir'],
            args_json['model_best'])
        checkpoint = torch.load(checkpoint_path)
        previous_epoch = checkpoint['epoch']
        previous_loss = checkpoint['validation_loss']
        previous_acc = checkpoint['validation_acc']
        if isinstance(model, CDCK2):
            model.update_flat_features(args_json['n_flat_features_per_frame'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info('\n\nResume from checkpoint: %s\n\n' % checkpoint_path)

    ###          ---------------------------Train and Validate---------------------------------
    best_acc = previous_acc
    best_loss = previous_loss
    best_epoch = previous_epoch
    for epoch in range(previous_epoch, args_json['epochs'] + 1):
        epoch_timer = timer()

        train(args_json, model, device, train_loader, optimizer, epoch, args_json['batch_size'],
              args_json['learning_rate'])

        val_acc, val_loss = validation(model, args_json,device, validation_loader)
        logger.info(
            '-------------------------------\nValidation Epoch: {} \tAccuracy: {:.4f}\tLoss: {:.6f}\n------------------'.format(
                epoch, val_acc, val_loss))

        # Saving a General Checkpoint for Inference and/or Resuming Training
        if 'CP' in args_json['model_type'] and val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            snapshot(args_json['logging_dir'], run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            args_json['model_best'] = run_name + '-model_best.pth'  # update json file
            write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
            logger.info("Parameters saved as '%s.ini'! !!!!!!!!" % args_json['setting_name'])
            best_epoch = epoch + 1
        elif 'CP' not in args_json['model_type'] and val_loss< best_loss:
            best_loss=min(val_loss,best_loss)
            snapshot(args_json['logging_dir'], run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            args_json['model_best'] = run_name + '-model_best.pth'  # update json file
            write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
            logger.info("Parameters saved as '%s.ini'! !!!!!!!!" % args_json['setting_name'])
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        logger.info(
            "#### End epoch {}/{}, elapsed time: {}".format(epoch, args_json['epochs'], end_epoch_timer - epoch_timer))

    ## end
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

    write_config(args_json, os.path.join(logging_dir, setting_name + '.ini'))
    logger.info("Parameters updated to '%s.ini'! !!!!!!!!" % args_json['setting_name'])
