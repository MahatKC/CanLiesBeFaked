import time, gc, os

import pandas as pd
from os.path import exists
from torch.utils.tensorboard import SummaryWriter

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag

from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import split_and_load

def save_to_csv(execution_id, learning_rate, lr_decay_strategy, optimizer, weight_decay, network, epochs, acc, val_acc):
    results_df = pd.DataFrame({
        'execution_id': [str(execution_id)],
        'learning_rate' : [str(learning_rate)],
        'lr_decay_strategy': [str(lr_decay_strategy)],
        'optimizer': [optimizer],
        'wd': [str(weight_decay)],
        'network': [network],
        'epochs': [str(epochs)],
        'training_acc': [str(acc)],
        'val_acc': [str(val_acc)]
    })

    if exists("hyperparameter_search.csv"):
        file_df = pd.read_csv("hyperparameter_search.csv")
        file_df = pd.concat([file_df,results_df], ignore_index=True)
        file_df.to_csv("hyperparameter_search.csv",index=False)
    else:
        results_df.to_csv("hyperparameter_search.csv",index=False)
    
    pass

def train_network(execution_id, ctx, network, epochs, lr_decay_epoch, optimizer, learning_rate, weight_decay):
    net = get_model(name=network, nclass=2)
    net.collect_params().reset_ctx(ctx)

    lr_decay = 0.1

    if optimizer=='sgd':
        optimizer_params = {'learning_rate': learning_rate, 'wd': weight_decay, 'momentum': 0.9} 
    else:
        #Using standard beta1, beta2 and epsilon for Adam and standard gamma1 and gamma2 for RMSProp
        optimizer_params = {'learning_rate': learning_rate, 'wd': weight_decay} 

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    train_metric = mx.metric.Accuracy()
    val_metric = mx.metric.Accuracy()
    test_metric = mx.metric.Accuracy()
    writer = SummaryWriter(log_dir='runs/run'+str(execution_id))

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
        
        #Get validation accuracy
        for i, batch in enumerate(val_data):
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

        print(f'[Epoch {epoch}] train={acc} val={val_acc} loss={train_loss/(i+1)} time: {time.time()-tic}')

    #Get test accuracy
    for i, batch in enumerate(test_data):
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        
        output = []
        for _, X in enumerate(data):
            X = X.reshape((-1,) + X.shape[2:])
            pred = net(X)
            output.append(pred)
        
        test_metric.update(label, output)

    name, test_acc = test_metric.get()
    print(f"Test acc={test_acc}")

    save_to_csv(execution_id, learning_rate, lr_decay_epoch, optimizer, weight_decay, network, epochs, acc, val_acc)

    writer.close()
    writer.flush()

    return test_acc

def train_5_fold_network(execution_id, ctx, network, epochs, lr_decay_epoch, optimizer, learning_rate, weight_decay):
    net = get_model(name=network, nclass=2)
    net.collect_params().reset_ctx(ctx)

    lr_decay = 0.1

    optimizer_params = {'learning_rate': learning_rate, 'wd': weight_decay, 'momentum': 0.9} 

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    train_metric = mx.metric.Accuracy()
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
        
        name, acc = train_metric.get()

        print(f'[Epoch {epoch}] train={acc} loss={train_loss/(i+1)} time: {time.time()-tic}')

    #Get test accuracy
    for i, batch in enumerate(test_data):
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        
        output = []
        for _, X in enumerate(data):
            X = X.reshape((-1,) + X.shape[2:])
            pred = net(X)
            output.append(pred)
        
        test_metric.update(label, output)

    name, test_acc = test_metric.get()
    print(f"Test acc={test_acc}")

    return test_acc

def load_train_val(length):
    transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = VideoClsCustom(root=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/Clips'),
                                setting=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/train.txt'),
                                train=True,
                                new_length=length,
                                video_loader=True,
                                slowfast = True,
                                use_decord=True,
                                transform=transform_train)
    val_dataset = VideoClsCustom(root=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/Clips'),
                                setting=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/val.txt'),
                                train=False,
                                new_length=length,
                                video_loader=True,
                                slowfast = True,
                                use_decord=True,
                                transform=transform_train)
                
    test_dataset = VideoClsCustom(root=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/Clips'),
                                setting=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/test.txt'),
                                train=False,
                                new_length=length,
                                video_loader=True,
                                slowfast = True,
                                use_decord=True,
                                transform=transform_train)

    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
    val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
    test_data = gluon.data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)

    return train_data, val_data, test_data

def load_folds(fold_index, length):
    transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    filenames = ['fold_0.txt', 'fold_1.txt', 'fold_2.txt', 'fold_3.txt', 'fold_4.txt']
    with open('Real-life_Deception_Detection_2016/train_with_fold_'+str(fold_index)+'_as_test.txt', 'w') as outfile:
        for fname in filenames:
            if fname == 'fold_'+str(fold_index)+'.txt':
                continue
            else:
                with open('Real-life_Deception_Detection_2016/'+fname) as infile:
                    for line in infile:
                        outfile.write(line)
                    outfile.write('\n')

    train_dataset = VideoClsCustom(root=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/Clips'),
                                setting=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/train_with_fold_'+str(fold_index)+'_as_test.txt'),
                                train=True,
                                new_length=length,
                                video_loader=True,
                                slowfast = True,
                                use_decord=True,
                                transform=transform_train)
                
    test_dataset = VideoClsCustom(root=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/Clips'),
                                setting=os.path.expanduser('/home/petcomp/TCC Mahat/Projeto/Real-life_Deception_Detection_2016/fold_'+str(fold_index)+'.txt'),
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

def train_5_fold(execution_id, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay):
    final_test_acc = 0
    execution_id *= 100
    for i in range(5):
        train_data, test_data = load_folds(i, 64)
        final_test_acc += train_5_fold_network(execution_id+i, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay)
        ctx[0].empty_cache()
        gc.collect() 
    with open('5FoldResults.txt', 'a') as results:
        results.write('Run '+str(execution_id)+" | Acuracia: "+str(final_test_acc/5)+'\n')
    print(f"ACURACIA FINAL: {final_test_acc/5}")

num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
per_device_batch_size = 1
num_workers = 1
batch_size = per_device_batch_size * num_gpus

hyperparameters = {
    'learning_rate' : [0.005, 0.001, 0.0005, 0.0001],
    'lr_decay_strategy': [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [40, 80, 100], [20, 40, 60, 80, 100], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200], [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300], [40, 80, 100, 300]],
    'optimizer': ['sgd', 'adam', 'rmsprop'],
    'wd': [0.001, 0.0001],
    'network': ['slowfast_4x16_resnet50_kinetics400', 'slowfast_4x16_resnet50_custom', 'slowfast_8x8_resnet50_kinetics400']
}

train_data, val_data, test_data = load_train_val(64)
network = 'slowfast_4x16_resnet50_kinetics400'
lr_decay_strategy = hyperparameters['lr_decay_strategy'][1]
chosen_optimizer = 'sgd'
weight_decay = 0.0001
num_epochs = 100
lr = 0.005
execution_id = 106
train_5_fold(execution_id, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay)
ctx[0].empty_cache()
gc.collect() 

lr = 0.001
execution_id = 108
lr_decay_strategy = hyperparameters['lr_decay_strategy'][0]
train_5_fold(execution_id, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay)
ctx[0].empty_cache()
gc.collect() 

lr = 0.0005
execution_id = 111
train_5_fold(execution_id, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay)
ctx[0].empty_cache()
gc.collect() 

lr = 0.001
execution_id = 117
weight_decay = 0.01
train_5_fold(execution_id, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay)
ctx[0].empty_cache()
gc.collect() 

lr = 0.001
execution_id = 165
weight_decay = 0.0001
lr_decay_strategy = hyperparameters['lr_decay_strategy'][4]
num_epochs = 200
train_5_fold(execution_id, ctx, network, num_epochs, lr_decay_strategy, chosen_optimizer, lr, weight_decay)
ctx[0].empty_cache()
gc.collect()


