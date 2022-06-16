import torch
import os
from torch import optim
from model import CCP
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from utils.evaluation import *
from utils.projection import *
from utils.seq_parser import *
from utils.dataset import *
from advertorch.context import ctx_noparamgrad_and_eval
import torchattacks
import torch.nn as nn

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class Solver(object):
    def __init__(self, config):
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.base_path = config.base_path
        self.model_path = config.model_path
        self.sample_path = config.sample_path
        self.logs_path = config.logs_path
        self.validation_path = config.validation_path
        self.num_epochs = config.num_epochs
        self.log_step = config.log_step
        self.lr = config.lr
        self.lr_update_mode = config.lr_update_mode
        self.lr_schedule = config.lr_schedule

        self.save_every = config.save_every
        self.seed = config.seed
        self.bound_U = config.bound_U
        self.bound_C = config.bound_C
        self.project = config.project
        self.project_frequency = config.project_frequency
        self.pretrain = config.pretrain
        self.pretrain_epochs = config.pretrain_epochs
        self.hidden_size = config.hidden_size
        self.n_channels = config.n_channels
        self.valid_ratio = config.valid_ratio
        self.attack = config.attack
        self.attack_training = config.attack_training
        self.attack_pretrain = config.attack_pretrain
        self.epsilon_train = config.epsilon_train
        self.eps_iter_train = config.eps_iter_train
        self.nb_iter_train = config.nb_iter_train
        self.epsilon_test = config.epsilon_test
        self.eps_iter_test = config.eps_iter_test
        self.nb_iter_test = config.nb_iter_test

        self.lr_func = continuous_seq(**self.lr_schedule) if self.lr_schedule != None else None
        self.criterion = nn.CrossEntropyLoss()
        self.model = CCP(self.n_channels)
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)
        self.build_model()
        self.best_acc = 0
        self.best_epoch = 0
        self.best_model = self.model
        self.best_epoch_attack = 0
        self.best_acc_attack = 0
        self.best_acc_attack_corr_acc = 0
        self.best_model_attack = self.model

        self.adversary_train = torchattacks.PGD(self.model, eps=self.epsilon_train, alpha=self.eps_iter_train, steps=self.nb_iter_train)
        self.adversary_test = torchattacks.PGD(self.model, eps=self.epsilon_test, alpha=self.eps_iter_test, steps=self.nb_iter_test)

        if self.dataset == 'mnist':
            self.train_batches = (60000 - 1) // self.batch_size + 1
            self.train_loader, self.valid_loader, self.test_loader, classes = mnist(batch_size=self.batch_size,
                                                                     valid_ratio=self.valid_ratio)
        elif self.dataset == 'FashionMNIST':
            self.train_batches = (60000 - 1) // self.batch_size + 1
            self.train_loader, self.valid_loader, self.test_loader, classes = FashionMNIST(batch_size=self.batch_size,
                                                                     valid_ratio=self.valid_ratio)
        elif self.dataset == 'cifar10':
            self.train_batches = (60000 - 1) // self.batch_size + 1
            self.train_loader, self.valid_loader, self.test_loader, classes = CIFAR10(batch_size=self.batch_size,
                                                                     valid_ratio=self.valid_ratio)

    def build_model(self):
        torch.manual_seed(self.seed)
        self.model.apply(self.weights_init)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total params: {}".format(self.num_params))
        if torch.cuda.is_available():
            self.model.cuda()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self):
        acc_calculator = AverageCalculator()
        loss_calculator = AverageCalculator()
        if self.attack == True:
            acc_calculator_ = AverageCalculator()
            loss_calculator_ = AverageCalculator()

        for epoch in range(self.num_epochs):

            acc_calculator.reset()
            loss_calculator.reset()
            self.model.train()

            for i, (data, labels) in enumerate(self.train_loader):

                epoch_batch_idx = epoch + 1. / self.train_batches * i if self.lr_update_mode.lower() in ['batch',] else epoch

                # Update the learning rate
                lr_this_batch = self.lr_func(epoch_batch_idx)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_this_batch
                if i == 0:
                    print('Learning rate = %1.2e' % lr_this_batch)


                data = data.type(torch.FloatTensor)
                data = to_cuda(data)
                labels = to_cuda(labels)

                if self.attack_training == True and epoch >= self.attack_pretrain:
                    with ctx_noparamgrad_and_eval(self.model):
                        data = self.adversary_train(data, labels)
                        # data = fgsm_attack(self.model, self.criterion, data, labels, self.epsilon)

                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.criterion(pred, labels)
                acc = accuracy(pred, labels)
                loss.backward()
                self.optimizer.step()

                if self.project == True and self.pretrain == True and self.pretrain_epochs <= epoch and (i+1) % self.project_frequency == 0:
                    with torch.no_grad():
                        params = list(self.model.named_parameters())
                        params[0][1].data = torch.reshape(params[0][1].data, (self.n_channels, 1 * 3 * 3))
                        params[2][1].data = torch.reshape(params[2][1].data, (self.n_channels, 16 * 7 * 7))
                        params[3][1].data = torch.reshape(params[3][1].data, (self.n_channels, 16 * 7 * 7))
                        params[4][1].data = torch.reshape(params[4][1].data, (self.n_channels, 16 * 7 * 7))
                        params[5][1].data = torch.reshape(params[5][1].data, (self.n_channels, 16 * 7 * 7))
                        temp_tensor = torch.cat((params[0][1].data, params[1][1].data.unsqueeze(1)), 1)
                        temp_tensor = proj_inf_matrix_norm(temp_tensor, self.bound_U)
                        params[0][1].data = temp_tensor[:, :-1]
                        params[1][1].data = temp_tensor[:, -1]
                        if self.bound_U < 1:
                            params[2][1].data = proj_inf_matrix_norm(params[2][1].data, 1)
                        else:
                            params[2][1].data = proj_inf_matrix_norm(params[2][1].data, self.bound_U)
                        params[3][1].data = proj_inf_matrix_norm(params[3][1].data, self.bound_U)
                        params[4][1].data = proj_inf_matrix_norm(params[4][1].data, self.bound_U)
                        params[5][1].data = proj_inf_matrix_norm(params[5][1].data, self.bound_U)
                        params[0][1].data = torch.reshape(params[0][1].data, (self.n_channels, 1, 3, 3))
                        params[2][1].data = torch.reshape(params[2][1].data, (self.n_channels, 16, 7, 7))
                        params[3][1].data = torch.reshape(params[3][1].data, (self.n_channels, 16, 7, 7))
                        params[4][1].data = torch.reshape(params[4][1].data, (self.n_channels, 16, 7, 7))
                        params[5][1].data = torch.reshape(params[5][1].data, (self.n_channels, 16, 7, 7))
                        temp_tensor = torch.cat((params[6][1].data, params[7][1].data.unsqueeze(1)), 1)
                        temp_tensor = proj_inf_matrix_norm(temp_tensor, self.bound_C)
                        params[6][1].data = temp_tensor[:, :-1]
                        params[7][1].data = temp_tensor[:, -1]

                loss_calculator.update(loss.item(), data.size(0))
                acc_calculator.update(acc.item(), data.size(0))

            loss_this_epoch = loss_calculator.average
            acc_this_epoch = acc_calculator.average
            print('Train loss / acc after epoch %d: %.4f / %.2f%%' % ((epoch + 1), loss_this_epoch, acc_this_epoch * 100.))

            loss_calculator.reset()
            acc_calculator.reset()
            if self.attack == True:
                loss_calculator_.reset()
                acc_calculator_.reset()

            self.model.eval()

            for i, (data_test, labels_test) in enumerate(self.test_loader):
                data_test = data_test.type(torch.FloatTensor)
                data_test = to_cuda(data_test)
                labels_test = to_cuda(labels_test)

                pred_test = self.model(data_test)
                loss_test = self.criterion(pred_test, labels_test)
                acc_test = accuracy(pred_test, labels_test)
                loss_calculator.update(loss_test.item(), data_test.size(0))
                acc_calculator.update(acc_test.item(), data_test.size(0))

                if self.attack == True:
                    with ctx_noparamgrad_and_eval(self.model):
                        data_test = self.adversary_test(data_test, labels_test)
                        pred_test = self.model(data_test)
                        loss_test = self.criterion(pred_test, labels_test)
                        acc_test = accuracy(pred_test, labels_test)

                        loss_calculator_.update(loss_test.item(), data_test.size(0))
                        acc_calculator_.update(acc_test.item(), data_test.size(0))


            loss_this_epoch = loss_calculator.average
            acc_this_epoch = acc_calculator.average
            print('Test loss / acc after epoch %d: %.4f / %.2f%%' % ((epoch + 1), loss_this_epoch, acc_this_epoch * 100.))

            if self.attack == True:
                attack_loss_this_epoch = loss_calculator_.average
                attack_acc_this_epoch = acc_calculator_.average
                print('Attack test loss / acc after epoch %d: %.4f / %.2f%%' % ((epoch + 1), attack_loss_this_epoch, attack_acc_this_epoch * 100.))


            if epoch >= self.pretrain_epochs and epoch >= self.attack_pretrain and acc_this_epoch >= self.best_acc:
                self.best_acc = acc_this_epoch
                self.best_epoch = epoch
                self.best_model = self.model

            if self.attack == True and epoch >= self.pretrain_epochs and epoch >= self.attack_pretrain and attack_acc_this_epoch >= self.best_acc_attack:
                self.best_acc_attack = attack_acc_this_epoch
                self.best_epoch_attack = epoch
                self.best_acc_attack_corr_acc = acc_this_epoch
                self.best_model_attack = self.model

            if (epoch + 1) % self.save_every == 0:
                CCP_model_path = os.path.join(self.model_path, 'model-{}.pkl'.format(epoch+1))
                torch.save(self.model.state_dict(), CCP_model_path)

        CCP_model_path = os.path.join(self.model_path, 'model-best.pkl')
        torch.save(self.best_model.state_dict(), CCP_model_path)
        print('Best acc is after epoch %d: %.2f%%' % ((self.best_epoch + 1), self.best_acc * 100.))
        if self.attack == True:
            CCP_model_path = os.path.join(self.model_path, 'model-best-acc.pkl')
            torch.save(self.best_model_attack.state_dict(), CCP_model_path)
            print('Best attack acc is after epoch %d: %.2f%% its corr acc %.2f%%' % ((self.best_epoch_attack + 1), self.best_acc_attack * 100., self.best_acc_attack_corr_acc * 100.))