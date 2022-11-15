import time, gc, os

import pandas as pd
import numpy as np
from os.path import exists
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag

from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import split_and_load

upper_dir = Path(os.getcwd()).parents[0]

def save_to_csv(execution_id, learning_rate, decay_strategy, optimizer, momentum, wd, network, epochs, acc, val_acc):
    results_df = pd.DataFrame({
        'execution_id': [str(execution_id)],
        'learning_rate' : [str(learning_rate)],
        'decay_strategy': [str(decay_strategy)],
        'optimizer': [optimizer],
        'momentum': [momentum],
        'wd': [str(wd)],
        'network': [network],
        'epochs': [str(epochs)],
        'training_acc': [str(acc)],
        'val_acc': [str(val_acc)]
    })

    if exists("hyperparameter_search_both_datasets.csv"):
        file_df = pd.read_csv("hyperparameter_search_both_datasets.csv")
        file_df = pd.concat([file_df,results_df], ignore_index=True)
        file_df.to_csv("hyperparameter_search_both_datasets.csv",index=False)
    else:
        results_df.to_csv("hyperparameter_search_both_datasets.csv",index=False)

def train_network(is_hyperparam_search, network, run_tag, train_data, test_data, training_params):
    execution_id = training_params[0]
    ctx = training_params[1]
    epochs = training_params[2]
    lr_decay_epoch = training_params[3]
    optimizer = training_params[4]
    learning_rate = training_params[5]
    wd = training_params[6]
    momentum= training_params[7]
    
    net = get_model(name=network, nclass=2)
    net.collect_params().reset_ctx(ctx)

    lr_decay = 0.1

    if optimizer=='sgd':
        optimizer_params = {'learning_rate': learning_rate, 'wd': wd, 'momentum': momentum} 
    else:
        #Using standard beta1, beta2 and epsilon for Adam and standard gamma1 and gamma2 for RMSProp
        optimizer_params = {'learning_rate': learning_rate, 'wd': wd} 

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    train_metric = mx.metric.Accuracy()
    if is_hyperparam_search:
        val_metric = mx.metric.Accuracy()
        writer = SummaryWriter(log_dir='runs'+run_tag+'/run'+str(execution_id))
    else:
        test_metric = mx.metric.Accuracy()
    
    lr_decay_count = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        train_loss = 0

        # Learning rate decay
        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        # Loop through each batch of training data
        for i, batch in enumerate(train_data):
            # Extract data and label
            data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            # AutoGrad
            with ag.record():
                output = []
                for _, X in enumerate(data):
                    X = X.reshape((-1,) + X.shape[2:])
                    pred = net(X)
                    output.append(pred)
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

            # Backpropagation
            for l in loss:
                l.backward()

            # Optimize
            trainer.step(batch_size)

            # Update metrics
            train_loss += sum([l.mean().asscalar() for l in loss])
            train_metric.update(label, output)
        
        if is_hyperparam_search:
            #Get validation accuracy
            for i, batch in enumerate(test_data):
                data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                
                output = []
                for _, X in enumerate(data):
                    X = X.reshape((-1,) + X.shape[2:])
                    pred = net(X)
                    output.append(pred)
                
                val_metric.update(label, output)

            name, acc = train_metric.get()
            name, val_acc = val_metric.get()

            # Update Tensorboard
            writer.add_scalar('Accuracy/train', acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

            if epoch%20==0:
                print(f'[Epoch {epoch}] train={acc} val={val_acc} loss={train_loss/(i+1)} time: {time.time()-tic}')


    if is_hyperparam_search:
        save_to_csv(execution_id, learning_rate, lr_decay_epoch, optimizer, momentum, wd, network, epochs, acc, val_acc)

        writer.close()
        writer.flush()

        test_acc, cm = 0, 0
    else:
        all_labels = []
        all_outputs = []

        #Get test accuracy
        for i, batch in enumerate(test_data):
            data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            
            output = []
            for _, X in enumerate(data):
                X = X.reshape((-1,) + X.shape[2:])
                pred = net(X)
                output.append(pred)
            
            for l in label:
                all_labels.extend(l.asnumpy().tolist())

            for o in output[0]:
                all_outputs.append(np.argmax(o.asnumpy()))
            
            test_metric.update(label, output)

        cm = confusion_matrix(all_labels, all_outputs)

        name, test_acc = test_metric.get()
        print(f"Train acc= {acc} Test acc={test_acc}")

    return test_acc, cm

def load_data(train_vids_path, train_config_path, test_vids_path, test_config_path, length):
    transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = VideoClsCustom(root=train_vids_path,
                                setting=train_config_path,
                                train=True,
                                new_length=length,
                                video_loader=True,
                                slowfast = True,
                                use_decord=True,
                                transform=transform_train)
                
    test_dataset = VideoClsCustom(root=test_vids_path,
                                setting=test_config_path,
                                train=False,
                                new_length=length,
                                video_loader=True,
                                slowfast = True,
                                use_decord=True,
                                transform=transform_train)

    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
    test_data = gluon.data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)

    return train_data, test_data

def get_network(slowfast):
    if slowfast=='8x8':
        network = 'slowfast_8x8_resnet50_kinetics400'
        length = 128
    else:
        network = 'slowfast_4x16_resnet50_kinetics400'
        length = 64

    return network, length

def hyperparam_search(experiment_tag, is_hyperparam_search, network, length, training_params):
    if experiment_tag=='A':
        rlt_clips = 'Real-life_Deception_Detection_2016/Clips'
        rlt_train = 'Real-life_Deception_Detection_2016/train_with_fold_0_as_test.txt'
        rlt_test = 'Real-life_Deception_Detection_2016/fold_0_as_test.txt'
        train_data, test_data = load_data(rlt_clips, rlt_train, rlt_clips, rlt_test, length)
        train_network(is_hyperparam_search, network, 'RLT', train_data, test_data, training_params)
    elif experiment_tag=='B':
        bol_clips = 'Box of Lies Vids/Clips'
        bol_train = 'Box of Lies Vids/train_with_fold_0_as_test.txt'
        bol_test = 'Box of Lies Vids/fold_0_as_test.txt'
        train_data, test_data = load_data(bol_clips, bol_train, bol_clips, bol_test, length)
        train_network(is_hyperparam_search, network, 'BoL', train_data, test_data, training_params)
    elif experiment_tag=='E':
        both_clips = 'Both Datasets/Clips'
        both_train = 'Both Datasets/train_with_fold_0_as_test.txt'
        both_test = 'Both Datasets/fold_0_as_test.txt'
        train_data, test_data = load_data(both_clips, both_train, both_clips, both_test, length)
        train_network(is_hyperparam_search, network, 'Both', train_data, test_data, training_params)  
    
    ctx[0].empty_cache()
    gc.collect() 

def cross_testing(experiment_tag, is_hyperparam_search, network, length, training_params):
    rlt_clips = 'Real-life_Deception_Detection_2016/Clips'
    rlt_config = 'Real-life_Deception_Detection_2016/whole_rlt.txt'
    bol_clips = 'Box of Lies Vids/Clips'
    bol_config = 'Box of Lies Vids/whole_bol.txt'

    if experiment_tag=='C':
        train_data, test_data = load_data(rlt_clips, rlt_config, bol_clips, bol_config, length)
        with open('Results_RLT_to_BoL.txt', 'a') as results:
            results.write('\nRun '+str(execution_id)+"\n")
            test_acc, cm = train_network(is_hyperparam_search, network, '', train_data, test_data, training_params)  
            results.write("Acuracia: "+str(test_acc)+'\n')
            results.write(str(cm))
            results.write('----------------------------------')
    else:
        train_data, test_data = load_data(bol_clips, bol_config, rlt_clips, rlt_config, length)
        with open('Results_BoL_to_RLT.txt', 'a') as results:
            results.write('\nRun '+str(execution_id)+"\n")
            test_acc, cm = train_network(is_hyperparam_search, network, '', train_data, test_data, training_params)  
            results.write("Acuracia: "+str(test_acc)+'\n')
            results.write(str(cm))
            results.write('---------------------------------')

def test_5_fold(fold_params, clips_path, config_path, run_tag):
    is_hyperparam_search = fold_params[0]
    network = fold_params[1]
    length = fold_params[2]
    training_params = fold_params[3]
    clips = clips_path + 'Clips'
    final_test_acc = 0
    
    with open('5FoldResults'+run_tag+'.txt', 'a') as results:
        results.write('\nRun '+str(execution_id)+"\n")
        for i in range(5):
            train_config = config_path + 'train_with_fold_' + str(i) + '_as_test.txt'
            test_config = config_path + 'fold_' + str(i) + '.txt'
            
            train_data, test_data = load_data(clips, train_config, clips, test_config, length)
            test_acc, cm = train_network(is_hyperparam_search, network, '', train_data, test_data, training_params)
            
            #ctx[0].empty_cache()
            gc.collect()

            final_test_acc += test_acc
            results.write('------Fold '+str(i)+" | Acuracia: "+str(test_acc)+'\n')
            results.write(str(cm))
            results.write('\n')

        results.write("ACURACIA FINAL: "+str(final_test_acc/5)+'\n')
    print(f"ACURACIA FINAL: {final_test_acc/5}")

def unpack_test_5_fold(experiment_tag, is_hyperparam_search, network, length, training_params):
    fold_params = [is_hyperparam_search, network, length, training_params]
    if experiment_tag == 'A':
        test_5_fold(fold_params, 'Real-life_Deception_Detection_2016/', 'Real-life_Deception_Detection_2016/', 'RLT')
    elif experiment_tag == 'B':
        test_5_fold(fold_params, 'Box of Lies Vids/', 'Box of Lies Vids/', 'BoL')
    elif experiment_tag == 'E':
        test_5_fold(fold_params, 'Both Datasets/', 'Both Datasets/', 'Both')
    elif experiment_tag == 'F':
        test_5_fold(fold_params, 'Both Datasets/', 'Augmented RLT/', 'RLTAugmented')
    else:
        test_5_fold(fold_params, 'Both Datasets/', 'Augmented BoL/', 'BoLAugmented')

def run(experiment_tag, is_hyperparam_search, slowfast, training_params):
    network, length = get_network(slowfast)
    if is_hyperparam_search:
        hyperparam_search(experiment_tag, is_hyperparam_search, network, length, training_params)
    else:
        if experiment_tag=='C' or experiment_tag=='D':
            cross_testing(experiment_tag, is_hyperparam_search, network, length, training_params)
        else:
            unpack_test_5_fold(experiment_tag, is_hyperparam_search, network, length, training_params)

num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
per_device_batch_size = 1
num_workers = 1
batch_size = per_device_batch_size * num_gpus

#HYPERPARAMETERS - can be set individually or used in lists to be iterated on
execution_id = 1
slowfast = '4x16'
epochs = 200
decay_strategy = [500]
optimizer = 'adam'
lr = 0.00001
wd = 0.0001
momentum = 0.9

experiment_tag = 'F'
# A - RLT-RLT
# B - BoL-BoL
# C - RLT-BoL
# D - BoL-RLT
# E - RLT+BoL-RLT+BoL
# F - RLT+BoL-RLT
# G - RLT+BoL-BoL
is_hyperparam_search = False
training_params = [execution_id, ctx, epochs, decay_strategy, optimizer, lr, wd, momentum]

run(experiment_tag, is_hyperparam_search, slowfast, training_params)