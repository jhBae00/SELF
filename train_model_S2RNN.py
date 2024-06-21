import argparse

import numpy as np
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
# import model.ARMBAND_bias2 as ARMBANDGNN
# import model.SimpleAttention as ARMBANDGNN
# import model.TCN as ARMBANDGNN
#import model.ARMBAND_RAW as ARMBANDGNN
#import model.S2RNN as S2RNN
import model.ARMBAND_bias2_jhbae_last_raw as ARMBANDGNN_last


import model.TCN as TCN

torch.manual_seed(0)
np.random.seed(0)
import warnings
warnings.filterwarnings(action='ignore')

total_loss = []
total_acc = []
valid_loss = []
valid_acc = []


def add_args(parser):
    parser.add_argument('--gpu', type=int, default=1, metavar='N', help='GPU index.')
    
    parser.add_argument('--epoch', type=int, default=150, metavar='N',
                        help='number of training')
    parser.add_argument('--epoch_pre', type=int, default=300, metavar='N',
                    help='number of training')
    parser.add_argument('--dataset', default='skku', metavar='N', help='dataset type: skku, nina, nina_18')


    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='insert batch size for training(default 128)')

    parser.add_argument('--precision', type=float, default=1e-6, metavar='N',
                        help='reducing learning rate when a metric has stopped improving(default = 0.0000001')

    parser.add_argument('--channel',default='[24, 16, 8, 4]',metavar='N', help=' 3 channel')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='probability of elements to be zero')
    parser.add_argument('--type', type=int, default=2, metavar='N',
                        help='0: GNN, 1: concat version, 2: GAT version')
    parser.add_argument('--indi', type=int, default=1, metavar='N',
                        help='0: total, 1: indi')
    parser.add_argument('--data', default="['NM_정희수2_sep1_data230724_raw.npy', 'NM_정희수2_sep2_data230724_raw.npy', 'NM_정희수2_sep3_data230724_raw.npy','NM_정희수2_sep4_data230724_raw.npy']", metavar='N',
                        help='name of dataset, nina5_data_xshit.npy, new_data_36.npy, evaluation_example.npy')
    parser.add_argument('--label', default="['NM_정희수2_sep1_label230724_raw.npy', 'NM_정희수2_sep2_label230724_raw.npy', 'NM_정희수2_sep3_label230724_raw.npy', 'NM_정희수2_sep4_label230724_raw.npy']", metavar='N',
                        help='name of label nina5_label.npy, new_label_36.npy, evaluation_labels.npy')
    parser.add_argument('--cand_num', type=int, default=4, metavar='N',
                        help='number of candidates for each dataset, 10, 36, 17')
    parser.add_argument('--load_data', default='./utils/saved_model/4th_sep4.pt', metavar='N',
                        help='saved model name(no duplicate)')
    parser.add_argument('--num_label', type=int, default= 18, metavar='N',
                        help = 'numbe of label')
    parser.add_argument('--channel_electrode', type=int, default=8, metavar='N')

    args = parser.parse_args()

    return args


def scramble(examples, labels):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)

    new_labels, new_examples = [], []
    for i in random_vec:
        new_labels.append(labels[i])
        new_examples.append(examples[i])

    return new_examples, new_labels


def train_model_w_da(pretrain_model, model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision, optimizer2):
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20
    hundred = False

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    outputs = model(inputs) # forward
                    # print(outputs[0][-1].shape)
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:# phase == 'val'
                    model.eval()
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 53))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(1):#range(20)
                        outputs = model(inputs)
                        labels = labels.long()
                        loss = criterion(outputs, labels)
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions.cpu() == labels.data.cpu())
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': #'val':  #TODO changed to train
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    print("stopped because of 100%")
                    hundred = True

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break
    # domain_adaptation time
    for epoch in range(50):
        epoch_start = time.time()
        print('domain Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10, "^ㅁ^bbbbb")
        running_loss = 0.
        running_corrects = 0
        total = 0
        for i, data in enumerate(dataloaders['pretrain']):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            optimizer2.zero_grad()
            optimizer.zero_grad()
            model.train()
            pretrain_model.train()
            outputs = pretrain_model(inputs)
            outputs = model(outputs)  # forward
            # print(outputs[0][-1].shape)
            _, predictions = torch.max(outputs.data, 1)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer2.step()
            loss = loss.item()

    #####
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load(args.load_data)
    model.load_state_dict(model_weights)
    model.eval()
    return pretrain_model, model, num_epochs


def adapt_mlp_nina_w_da(model_basic, model_da, criterion, optimizer_basic, optimizer, scheduler, dataloaders, num_epochs, precision):
    
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('domain adaptation Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model_da.train(True)  # Set model to training mode
            else:
                model_da.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                # inputs = torch.nn.functional.normalize(inputs)
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                optimizer.zero_grad()
                optimizer_basic.zero_grad()

                if phase == 'train':
                    model_da.train()
                    model_basic.train()
                    '''
                    for param in model_basic.parameters():
                        param.requires_grad = False
                    
                    '''
                    outputs_da = model_da(inputs) # forward
                    outputs = model_basic(outputs_da)
                    # print(outputs[0][-1].shape)
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:# phase == 'val'
                    model_da.eval()
                    model_basic.eval()
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 53))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(1):#range(20)
                        outputs_da = model_da(inputs) # forward
                        outputs = model_basic(outputs_da)
                        labels = labels.long().cpu()
                        loss = criterion(outputs.cpu(), labels)
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': #'val':  #TODO changed to train
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    #torch.save(model_da.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    print("stopped because of 100%")
                    hundred = True

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    #model_weights = torch.load(args.load_data)
    #model.load_state_dict(model_weights)
    model_da.eval()
    return model_da, num_epochs


def fit_freeze_w_da(args, examples_training, labels_training):
    accuracy_test0, accuracy_test1 = [], []
    X_fine_tune_train, Y_fine_tune_train = [], []
    X_fine_tune_pretrain, Y_fine_tune_pretrain = [], []
    X_fine_tune_test, Y_fine_tune_test = [], []
    if(args.dataset == 'skku'):
        for dataset_index in range(0, 2):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_train.extend(examples_training[label_index][example_index])
                            Y_fine_tune_train.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 3):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_pretrain.extend(examples_training[label_index][example_index])
                            Y_fine_tune_pretrain.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(3, 4):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                        X_fine_tune_test.extend(examples_training[label_index][example_index])
                        Y_fine_tune_test.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
    else:
        for dataset_index in range(0, 2):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_train.extend(examples_training[label_index][example_index])
                            Y_fine_tune_train.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 4):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_pretrain.extend(examples_training[label_index][example_index])
                            Y_fine_tune_pretrain.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(4, 6):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                        X_fine_tune_test.extend(examples_training[label_index][example_index])
                        Y_fine_tune_test.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        


    X_fine_tunning, Y_fine_tunning = scramble(X_fine_tune_train, Y_fine_tune_train)
    X_fine_pretrain, Y_fine_pretrain = scramble(X_fine_tune_pretrain, Y_fine_tune_pretrain)
    X_test_0, Y_test_0 = scramble(X_fine_tune_test, Y_fine_tune_test)


    valid_examples = X_fine_tunning[0:int(len(X_fine_tunning) * 0.2)]
    labels_valid = Y_fine_tunning[0:int(len(Y_fine_tunning) * 0.2)]
    X_fine_tune = X_fine_tunning[int(len(X_fine_tunning) * 0.2):]
    Y_fine_tune = Y_fine_tunning[int(len(Y_fine_tunning) * 0.2):]
    # X_test_0 = X_fine_tunning[int(len(X_fine_tunning)*0.1):int(len(X_fine_tunning)*0.2)]
    # Y_test_0 = Y_fine_tunning[int(len(X_fine_tunning)*0.1):int(len(X_fine_tunning)*0.2)]
    
    X_fine_pretrain_valid = X_fine_pretrain[0:int(len(X_fine_tunning) * 0.2)]
    Y_fine_pretrain_valid = Y_fine_pretrain[0:int(len(Y_fine_tunning) * 0.2)]
    X_fine_pretrain = X_fine_pretrain[int(len(X_fine_tunning) * 0.2):]
    Y_fine_pretrain = Y_fine_pretrain[int(len(Y_fine_tunning) * 0.2):]
    

    print("total data size :", len(X_fine_tune_train), np.shape(np.array(X_fine_tune_train)))

    X_fine_tune = torch.from_numpy(np.array(X_fine_tune, dtype=np.float32))
    X_fine_pretrain = torch.from_numpy(np.array(X_fine_pretrain, dtype=np.float32))
    #X_fine_tune = torch.transpose(X_fine_tune, 1, 2)

    print("train data :", np.shape(np.array(X_fine_tune)))
    Y_fine_tune = torch.from_numpy(np.array(Y_fine_tune, dtype=np.float32))
    Y_fine_pretrain = torch.from_numpy(np.array(Y_fine_pretrain, dtype=np.float32))
    
    X_fine_pretrain_valid = torch.from_numpy(np.array(X_fine_pretrain_valid, dtype=np.float32))
    Y_fine_pretrain_valid = torch.from_numpy(np.array(Y_fine_pretrain_valid, dtype=np.float32))
    
    
    valid_examples = torch.from_numpy(np.array(valid_examples, dtype=np.float32))
    #valid_examples = torch.transpose(valid_examples, 1, 2)
    print("valid data :", np.shape(np.array(valid_examples)))
    labels_valid = torch.from_numpy(np.array(labels_valid, dtype=np.float32))
    # dimension setting
    
    X_test_0 = torch.from_numpy(np.array(X_test_0, dtype=np.float32))
    #X_test_0 = torch.transpose(X_test_0, 1, 2)
    Y_test_0 = torch.from_numpy(np.array(Y_test_0, dtype=np.float32))


    # dataset
    # X_fine_tune = torch.unsqueeze(X_fine_tune, dim=2)
    # valid_examples = torch.unsqueeze(valid_examples, dim=2)
    # X_test_0 = torch.unsqueeze(X_test_0, dim=2)
    train_fine = TensorDataset(X_fine_pretrain, Y_fine_pretrain) #for domain adptation
    valid_fine = TensorDataset(X_fine_pretrain_valid, Y_fine_pretrain_valid)
    train = TensorDataset(X_fine_tune, Y_fine_tune)
    valid = TensorDataset(valid_examples, labels_valid)
    test_0 = TensorDataset(X_test_0, Y_test_0)
    print(torch.unique(Y_fine_tune))
    print(torch.unique(labels_valid))
    print(torch.unique(Y_test_0))
    # data loading
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=1, shuffle=False)
    pretrain_loader = torch.utils.data.DataLoader(train_fine, batch_size=args.batch_size, shuffle=True)

    train_loader_fine = torch.utils.data.DataLoader(train_fine, batch_size=args.batch_size, shuffle=True)
    valid_loader_fine = torch.utils.data.DataLoader(valid_fine, batch_size=args.batch_size, shuffle=True)


    
    '''
    #희수님 방식
    stgcn = ARMBANDGNN_last.S2RNN(args.channel_electrode, args.num_label).cuda(args.gpu)
    adaptation_model = ARMBANDGNN_last.Domain_adaptation_mlp(model, args.channel_electrode).cuda(args.gpu)

    precision = 1e-8
    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(stgcn.parameters(), lr=args.lr) #lr=args.lr)  # lr=0.0404709 lr=args.lr
    optimizer2 = optim.Adam(pretrain_model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.4, patience=5,
                                                     verbose=True, eps=precision)
    #training
    pretrain_model, model, num_epoch = train_model(pretrain_model, stgcn, criterion, optimizer, scheduler,\
                                   {"train": train_loader, "val": valid_loader, 'pretrain': pretrain_loader}, args.epoch, args.precision, optimizer2)
    pretrain_model.eval()
    model.eval()
    
    '''
    
    stgcn = ARMBANDGNN_last.S2RNN(args.channel_electrode, args.num_label).cuda(args.gpu)
    
    precision = 1e-8
    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(stgcn.parameters(), lr=args.lr) #lr=args.lr)  # lr=0.0404709 lr=args.lr
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.4, patience=5,
                                                     verbose=True, eps=precision)
    #training
    model_basic, num_epoch = train_model(stgcn, criterion, optimizer, scheduler, {"train": train_loader, "val": valid_loader}, args.epoch, precision)
    
    

    adaptation_model = ARMBANDGNN_last.Domain_adaptation_pre(args.channel_electrode).cuda(args.gpu)
    optimizer_ft = optim.Adam(adaptation_model.parameters(), lr=args.lr) 
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', factor=.4, patience=5,
                                                        verbose=True, eps=precision)
    model_da, num_epoch = adapt_mlp_nina_w_da(model_basic, adaptation_model, criterion, optimizer, optimizer_ft, scheduler_ft,\
                                    {"train": train_loader_fine, "val": valid_loader_fine}, args.epoch_pre, args.precision)
    
    
    all_dict = dict()
    correct_dict = dict()
    for i in range(args.num_label):
        all_dict[i] = 0
        correct_dict[i] = 0

    acc_perlabel = []
    # test : set_0
    total = 0
    correct_prediction_test_0 = 0
    time_list = []
    for k, data_test_0 in enumerate(test_0_loader):
        start_time = time.time()
        inputs_test_0, ground_truth_test_0 = data_test_0
        # inputs_test_0 = torch.nn.functional.normalize(inputs_test_0)
        inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0).cuda(args.gpu), Variable(ground_truth_test_0).cuda(args.gpu)
        concat_input = inputs_test_0
        # for i in range(20): #input data 옆으로 20개 복사 concat
        #     concat_input = torch.cat([concat_input, inputs_test_0])
        outputs_da = model_da(concat_input)
        outputs_test_0 = model_basic(outputs_da)
        
        outputs_test_0 = outputs_test_0.cpu()
        _, predicted = torch.max(outputs_test_0.data, 1)
        all_dict[int(ground_truth_test_0)] += 1
        # print(predicted)
        if mode(predicted.cpu().numpy())[0][0] == ground_truth_test_0.data.cpu().numpy():
            correct_dict[int(ground_truth_test_0)] += 1
        correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
                                      ground_truth_test_0.data.cpu().numpy()).sum()
        total += ground_truth_test_0.size(0)
        end = time.time()
        time_list.append(end-start_time)
    accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
    print("ACCURACY TESƒT_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))


    # test : set_1
    for i in range(args.num_label):
        try:
            acc_perlabel.append(correct_dict[i]/all_dict[i])
            print("accuracy in %d : %f"%(i, correct_dict[i]/all_dict[i]))
        except:
            continue

    #result
    print("AVERAGE ACCURACY TEST 0:   %.3f" % np.array(accuracy_test0).mean())
    #print("average time: ", np.mean(time_list), time_list)
    return accuracy_test0,  num_epoch


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision):
    
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                # inputs = torch.nn.functional.normalize(inputs)
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    outputs = model(inputs) # forward
                    # print(outputs[0][-1].shape)
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:# phase == 'val'
                    model.eval()
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 53))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(1):#range(20)
                        outputs = model(inputs)
                        labels = labels.long().cpu()
                        loss = criterion(outputs.cpu(), labels)
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': #'val':  #TODO changed to train
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    print("stopped because of 100%")
                    hundred = True

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load(args.load_data)
    model.load_state_dict(model_weights)
    model.eval()
    return model, num_epochs


def adapt_mlp_nina(model, criterion, optimizer, scheduler, dataloaders, num_epochs, precision):
    
    since = time.time()
    best_loss = float('inf')
    patience = 60
    patience_increase = 20

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('domain adaptation Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10, "^ㅁ^bbbbb")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                # inputs = torch.nn.functional.normalize(inputs)
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
                optimizer.zero_grad()

                if phase == 'train':
                    model.train()
                    outputs = model(inputs) # forward
                    # print(outputs[0][-1].shape)
                    _, predictions = torch.max(outputs.data, 1)
                    labels = labels.long()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:# phase == 'val'
                    model.eval()
                    accumulated_predicted = Variable(torch.zeros(len(inputs), 53))
                    loss_intermediary = 0.
                    total_sub_pass = 0
                    for repeat in range(1):#range(20)
                        outputs = model(inputs)
                        labels = labels.long().cpu()
                        loss = criterion(outputs.cpu(), labels)
                        if loss_intermediary == 0.:
                            loss_intermediary = loss.item()
                        else:
                            loss_intermediary += loss.item()
                        _, prediction_from_this_sub_network = torch.max(outputs.data, 1)
                        accumulated_predicted[range(len(inputs)),
                                              prediction_from_this_sub_network.cpu().numpy().tolist()] += 1
                        total_sub_pass += 1
                    _, predictions = torch.max(accumulated_predicted.data, 1)
                    loss = loss_intermediary / total_sub_pass

                running_loss += loss
                running_corrects += torch.sum(predictions == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            if phase == 'val':
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            else:
                total_loss.append(epoch_loss)
                total_acc.append(epoch_acc)
            print('{} Loss: {:.8f} Acc: {:.8}'.format(
                phase, epoch_loss, epoch_acc))

            # earlystopping
            if phase == 'val': #'val':  #TODO changed to train
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), args.load_data)
                    patience = patience_increase + epoch
                if epoch_acc == 1:
                    print("stopped because of 100%")
                    hundred = True

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        if epoch > patience :
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model_weights = torch.load(args.load_data)
    model.load_state_dict(model_weights)
    model.eval()
    return model, num_epochs



def fit_freeze(args, examples_training, labels_training):
    accuracy_test0, accuracy_test1 = [], []
    X_fine_tune_train, Y_fine_tune_train = [], []
    X_fine_tune_pretrain, Y_fine_tune_pretrain = [], []
    X_fine_tune_test, Y_fine_tune_test = [], []
    if(args.dataset == 'skku'):
        for dataset_index in range(0, 2):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_train.extend(examples_training[label_index][example_index])
                            Y_fine_tune_train.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 3):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_pretrain.extend(examples_training[label_index][example_index])
                            Y_fine_tune_pretrain.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(3, 4):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                        X_fine_tune_test.extend(examples_training[label_index][example_index])
                        Y_fine_tune_test.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
    else:
        for dataset_index in range(0, 2):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_train.extend(examples_training[label_index][example_index])
                            Y_fine_tune_train.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(2, 4):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                            X_fine_tune_pretrain.extend(examples_training[label_index][example_index])
                            Y_fine_tune_pretrain.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        for dataset_index in range(4, 6):
            for label_index in range(len(labels_training)):
                if label_index == dataset_index:
                    for example_index in range(len(examples_training[label_index])):
                        # if (example_index < 208):
                        X_fine_tune_test.extend(examples_training[label_index][example_index])
                        Y_fine_tune_test.extend(labels_training[label_index][example_index])
            print("{}-th data set open~~~".format(dataset_index))
        


    X_fine_tunning, Y_fine_tunning = scramble(X_fine_tune_train, Y_fine_tune_train)
    X_fine_pretrain, Y_fine_pretrain = scramble(X_fine_tune_pretrain, Y_fine_tune_pretrain)
    X_test_0, Y_test_0 = scramble(X_fine_tune_test, Y_fine_tune_test)


    valid_examples = X_fine_tunning[0:int(len(X_fine_tunning) * 0.2)]
    labels_valid = Y_fine_tunning[0:int(len(Y_fine_tunning) * 0.2)]
    X_fine_tune = X_fine_tunning[int(len(X_fine_tunning) * 0.2):]
    Y_fine_tune = Y_fine_tunning[int(len(Y_fine_tunning) * 0.2):]
    # X_test_0 = X_fine_tunning[int(len(X_fine_tunning)*0.1):int(len(X_fine_tunning)*0.2)]
    # Y_test_0 = Y_fine_tunning[int(len(X_fine_tunning)*0.1):int(len(X_fine_tunning)*0.2)]
    
    X_fine_pretrain_valid = X_fine_pretrain[0:int(len(X_fine_tunning) * 0.2)]
    Y_fine_pretrain_valid = Y_fine_pretrain[0:int(len(Y_fine_tunning) * 0.2)]
    X_fine_pretrain = X_fine_pretrain[int(len(X_fine_tunning) * 0.2):]
    Y_fine_pretrain = Y_fine_pretrain[int(len(Y_fine_tunning) * 0.2):]
    

    print("total data size :", len(X_fine_tune_train), np.shape(np.array(X_fine_tune_train)))

    X_fine_tune = torch.from_numpy(np.array(X_fine_tune, dtype=np.float32))
    X_fine_pretrain = torch.from_numpy(np.array(X_fine_pretrain, dtype=np.float32))
    #X_fine_tune = torch.transpose(X_fine_tune, 1, 2)

    print("train data :", np.shape(np.array(X_fine_tune)))
    Y_fine_tune = torch.from_numpy(np.array(Y_fine_tune, dtype=np.float32))
    Y_fine_pretrain = torch.from_numpy(np.array(Y_fine_pretrain, dtype=np.float32))
    
    X_fine_pretrain_valid = torch.from_numpy(np.array(X_fine_pretrain_valid, dtype=np.float32))
    Y_fine_pretrain_valid = torch.from_numpy(np.array(Y_fine_pretrain_valid, dtype=np.float32))
    
    
    valid_examples = torch.from_numpy(np.array(valid_examples, dtype=np.float32))
    #valid_examples = torch.transpose(valid_examples, 1, 2)
    print("valid data :", np.shape(np.array(valid_examples)))
    labels_valid = torch.from_numpy(np.array(labels_valid, dtype=np.float32))
    # dimension setting
    
    X_test_0 = torch.from_numpy(np.array(X_test_0, dtype=np.float32))
    #X_test_0 = torch.transpose(X_test_0, 1, 2)
    Y_test_0 = torch.from_numpy(np.array(Y_test_0, dtype=np.float32))


    # dataset
    # X_fine_tune = torch.unsqueeze(X_fine_tune, dim=2)
    # valid_examples = torch.unsqueeze(valid_examples, dim=2)
    # X_test_0 = torch.unsqueeze(X_test_0, dim=2)
    train_fine = TensorDataset(X_fine_pretrain, Y_fine_pretrain) #for domain adptation
    valid_fine = TensorDataset(X_fine_pretrain_valid, Y_fine_pretrain_valid)
    train = TensorDataset(X_fine_tune, Y_fine_tune)
    valid = TensorDataset(valid_examples, labels_valid)
    test_0 = TensorDataset(X_test_0, Y_test_0)
    print(torch.unique(Y_fine_tune))
    print(torch.unique(labels_valid))
    print(torch.unique(Y_test_0))
    # data loading
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=True)
    test_0_loader = torch.utils.data.DataLoader(test_0, batch_size=1, shuffle=False)

    train_loader_fine = torch.utils.data.DataLoader(train_fine, batch_size=args.batch_size, shuffle=True)
    valid_loader_fine = torch.utils.data.DataLoader(valid_fine, batch_size=args.batch_size, shuffle=True)


    # stgcn = ARMBANDGNN.ARMBANDGNN(eval(args.channel), args.type, args.num_label) #our model
    #stgcn = ARMBANDGNN.ARMBANDGNN(args.channel_electrode, args.num_label) #Simple Attention
    # stgcn = ARMBANDGNN.TCN(args.channel_electrode, args.num_label, [32, 32, 64, 128], 3,
    #                        0.05)  # TCN         #CLSTM.ConvLSTM(1, 16, (3, 3), 1)#.cuda()
    # stgcn = CLSTM.ConvLSTM(1, 16, (3, 3), 1)#.cuda()
    #num_channels, channels, num_classes, input_feature_dim
    
    
    #stgcn = ARMBANDGNN_last.ARMBANDGNN_modified_rnn_raw(args.channel_electrode, eval(args.channel), args.num_label, 100).cuda(args.gpu)
    stgcn = ARMBANDGNN_last.S2RNN(args.channel_electrode, args.num_label).cuda(args.gpu)
    
    precision = 1e-8
    criterion = nn.NLLLoss(size_average=False)
    optimizer = optim.Adam(stgcn.parameters(), lr=args.lr) #lr=args.lr)  # lr=0.0404709 lr=args.lr
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.2, patience=5,
    #                                                  verbose=True, eps=args.precision) #학습이 개선되지 않을때 자동으로 학습률을 조절합니다.
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.00,  verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.4, patience=5,
                                                     verbose=True, eps=precision)
    #training
    
    model, num_epoch = train_model(stgcn, criterion, optimizer, scheduler, {"train": train_loader, "val": valid_loader}, args.epoch, precision)
    
    

    #adaptation_model = ARMBANDGNN_last.DA_mlp_whole(stgcn, args.channel_electrode, eval(args.channel)).cuda(args.gpu)
    adaptation_model = ARMBANDGNN_last.Domain_adaptation_mlp(model, args.channel_electrode).cuda(args.gpu)
    optimizer_ft = optim.Adam(adaptation_model.parameters(), lr=args.lr) 
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', factor=.4, patience=5,
                                                        verbose=True, eps=precision)
    model, num_epoch = adapt_mlp_nina(adaptation_model, criterion, optimizer_ft, scheduler_ft,\
                                    {"train": train_loader_fine, "val": valid_loader_fine}, args.epoch_pre, args.precision)
    
    
    
    
    
    model.eval()
    all_dict = dict()
    correct_dict = dict()
    for i in range(args.num_label):
        all_dict[i] = 0
        correct_dict[i] = 0

    acc_perlabel = []
    # test : set_0
    total = 0
    correct_prediction_test_0 = 0
    time_list = []
    for k, data_test_0 in enumerate(test_0_loader):
        start_time = time.time()
        inputs_test_0, ground_truth_test_0 = data_test_0
        # inputs_test_0 = torch.nn.functional.normalize(inputs_test_0)
        inputs_test_0, ground_truth_test_0 = Variable(inputs_test_0).cuda(args.gpu), Variable(ground_truth_test_0).cuda(args.gpu)
        concat_input = inputs_test_0
        # for i in range(20): #input data 옆으로 20개 복사 concat
        #     concat_input = torch.cat([concat_input, inputs_test_0])
        outputs_test_0 = model(concat_input)
        
        outputs_test_0 = outputs_test_0.cpu()
        _, predicted = torch.max(outputs_test_0.data, 1)
        all_dict[int(ground_truth_test_0)] += 1
        # print(predicted)
        if mode(predicted.cpu().numpy())[0][0] == ground_truth_test_0.data.cpu().numpy():
            correct_dict[int(ground_truth_test_0)] += 1
        correct_prediction_test_0 += (mode(predicted.cpu().numpy())[0][0] ==
                                      ground_truth_test_0.data.cpu().numpy()).sum()
        total += ground_truth_test_0.size(0)
        end = time.time()
        time_list.append(end-start_time)
    accuracy_test0.append(100 * float(correct_prediction_test_0) / float(total))
    print("ACCURACY TESƒT_0 FINAL : %.3f %%" % (100 * float(correct_prediction_test_0) / float(total)))


    # test : set_1
    for i in range(args.num_label):
        try:
            acc_perlabel.append(correct_dict[i]/all_dict[i])
            print("accuracy in %d : %f"%(i, correct_dict[i]/all_dict[i]))
        except:
            continue

    #result
    print("AVERAGE ACCURACY TEST 0:   %.3f" % np.array(accuracy_test0).mean())
    #print("average time: ", np.mean(time_list), time_list)
    return accuracy_test0,  num_epoch


if __name__ == "__main__":
    # loading...
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if (args.dataset == 'nina'):
        data_file = "['NINA_1_sep1_data_raw.npy', 'NINA_1_sep2_data_raw.npy', 'NINA_1_sep3_data_raw.npy', 'NINA_1_sep4_data_raw.npy', 'NINA_1_sep5_data_raw.npy', 'NINA_1_sep6_data_raw.npy']"
        label_file = "['NINA_1_sep1_label_raw.npy', 'NINA_1_sep2_label_raw.npy', 'NINA_1_sep3_label_raw.npy', 'NINA_1_sep4_label_raw.npy', 'NINA_1_sep5_label_raw.npy', 'NINA_1_sep6_label_raw.npy']"
        examples_training = np.stack([np.load(f"./data/ninapro5/1번/{eval(data_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
        labels_training = np.stack([np.load(f"./data/ninapro5/1번/{eval(label_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
    elif (args.dataset == 'nina_18'):
        data_file = "['NINA_1_sep1_data_raw_18.npy', 'NINA_1_sep2_data_raw_18.npy', 'NINA_1_sep3_data_raw_18.npy', 'NINA_1_sep4_data_raw_18.npy', 'NINA_1_sep5_data_raw_18.npy', 'NINA_1_sep6_data_raw_18.npy']"
        label_file = "['NINA_1_sep1_label_raw_18.npy', 'NINA_1_sep2_label_raw_18.npy', 'NINA_1_sep3_label_raw_18.npy', 'NINA_1_sep4_label_raw_18.npy', 'NINA_1_sep5_label_raw_18.npy', 'NINA_1_sep6_label_raw_18.npy']"
        examples_training = np.stack([np.load(f"./data/ninapro5/1번/{eval(data_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
        labels_training = np.stack([np.load(f"./data/ninapro5/1번/{eval(label_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(6)])
    else:
        data_file = "['NM_정희수2_sep1_data230724_raw.npy', 'NM_정희수2_sep2_data230724_raw.npy', 'NM_정희수2_sep3_data230724_raw.npy','NM_정희수2_sep4_data230724_raw.npy']"
        label_file = "['NM_정희수2_sep1_label230724_raw.npy', 'NM_정희수2_sep2_label230724_raw.npy', 'NM_정희수2_sep3_label230724_raw.npy','NM_정희수2_sep4_label230724_raw.npy']"
        examples_training = np.concatenate([np.load(f"./data/skku/정희수2/{eval(data_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(4)])
        labels_training = np.concatenate([np.load(f"./data/skku/정희수2/{eval(label_file)[i]}", encoding="bytes", allow_pickle=True) for i in range(4)])
    #examples_training = np.concatenate([np.load(f"./data/skku/정희수2/{eval(args.data)[i]}", encoding="bytes", allow_pickle=True) for i in range(4)])
    #labels_training = np.concatenate([np.load(f"./data/skku/정희수2/{eval(args.label)[i]}", encoding="bytes", allow_pickle=True) for i in range(4)])
    #check_training = np.load(f"./data/skku/정희수2/{eval(args.data)[0]}", encoding="bytes", allow_pickle=True)
    #test_0, test_1 = [], []
    
    accuracy_test_0, num_epochs = fit_freeze_w_da(args, examples_training, labels_training)
    
