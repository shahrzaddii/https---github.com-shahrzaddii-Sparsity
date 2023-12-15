import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd


def normalise_data(x):
    mean_values = np.mean(x, axis=0)
    std_values = np.std(x, axis=0)
    std_values[std_values == 0] = 1.0
    x_normalized = (x - mean_values) / std_values

    return x_normalized, mean_values, std_values  

class StepActivation(nn.Module):
    def step(self, x):
        return torch.where(x >= 0, torch.tensor(1.0), torch.tensor(0.0))
    
class GeneralRegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, balancing_bias=False):
        super(GeneralRegressionNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1, bias=True)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.output = nn.Linear(hidden_size2, output_size, bias=False)
        self.balancing_bias = balancing_bias
        self.layer_dict = {1: 'hidden1.weight', 2: 'hidden2.weight', 3:'hidden3.weight', 4:'output.weight'}

   
    def forward(self, x):
        x = self.hidden1(x)
        x = torch.sigmoid(x)
        if self.balancing_bias is True:
            x = x - 1
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x
    
       
    def forward_binary(self, x):
        sa = StepActivation()
        x = self.hidden1(x)
        x = sa.step(x)
        if self.balancing_bias is True:
            x = x - 1
        x = self.hidden2(x)
        x = sa.step(x)
        x = self.output(x)
        return x

    
    def predict(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward(input)

        self.train()
        
        return output.numpy()
    
    def predict_binary(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward_binary(input)

        self.train()
        
        return output.numpy()
    
    def binarise_model(self, layer=2):
        for name, param in self.named_parameters():
            if name == self.layer_dict[layer]:
                param.data = torch.where(param.data >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

    

    def l1_penalty(self, l1_lambda):
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_lambda * l1_loss
    

    def binary_penalty(self, binary_lambda, layer=2):
        binary_loss = 0.0
        w_ = []
        for name, param in self.named_parameters():
            if name == self.layer_dict[layer]:
                penalised_param = torch.where(param < 0.5, torch.abs(param), torch.abs(param-1)).detach()
        binary_loss = torch.sum(penalised_param)
        return binary_lambda * binary_loss
    
    def l1_n_binary_penalty(self, l1_lambda, binary_lambda):
        l1_loss = 0
        for name, param in self.named_parameters():
            if name != self.layer_dict[2]:
                l1_loss += torch.sum(torch.abs(param))
        binary_loss = self.binary_penalty(binary_lambda)
        return binary_loss + l1_lambda*l1_loss
    



class GeneralLogisticNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, balancing_bias=False):
        super(GeneralLogisticNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1, bias=True)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        # self.hidden3 = nn.Linear(hidden_size2, hidden_size3, bias=False)
        self.output = nn.Linear(hidden_size2, output_size, bias=False)
        self.balancing_bias = balancing_bias
        self.layer_dict = {1: 'hidden1.weight', 2: 'hidden2.weight', 3:'hidden3.weight', 4:'output.weight'}

   
    def forward(self, x):
        x = self.hidden1(x)
        x = torch.sigmoid(x)
        if self.balancing_bias is True:
            x = x - 1
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
    

    def forward_binary(self, x):
        sa = StepActivation()
        x = self.hidden1(x)
        x = sa.step(x)
        if self.balancing_bias is True:
            x = x - 1
        x = self.hidden2(x)
        x = sa.step(x)
        x = self.output(x)
        x = sa.step(x)
        return x

    
    def predict(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward(input)

        self.train()
        
        return output.numpy()
    

    def predict_binary(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward_binary(input)

        self.train()
        
        return output.numpy()
    
    def binarise_model(self, layer=2):
        for name, param in self.named_parameters():
            if name == self.layer_dict[layer]:
                param.data = torch.where(param.data >= 0.5, torch.tensor(1.0), torch.tensor(0.0))


    def l1_penalty(self, l1_lambda):
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_lambda * l1_loss
    

    def binary_penalty(self, binary_lambda, layer=2):
        binary_loss = 0.0
        param = getattr(self, 'hidden2').weight
        binary_loss = torch.sum(torch.where(param < 0.5, torch.abs(param), torch.abs(param-1))).detach()
        return binary_lambda * binary_loss
    
    def l1_n_binary_penalty(self, l1_lambda, binary_lambda):
        l1_loss = 0
        for name, param in self.named_parameters():
            if name != self.layer_dict[2]:
                l1_loss += torch.sum(torch.abs(param))
        binary_loss = self.binary_penalty(binary_lambda)
        return binary_loss + l1_lambda*l1_loss
        



class ToTensor(Dataset):
    """ Input arg* :  X, y """
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples


class LossFunction():
    def __init__(self, criterion=None, mse=True, bce=False, l1_lambda=None, binary_lambda=None):
        self.l1_lambda = l1_lambda
        self.binary_lambda = binary_lambda
        self.mse = mse
        self.bce = bce
        self.criterion = criterion 

    def loss_criterion(self):
        if self.mse == True:
            self.loss = nn.MSELoss()
        if self.bce == True:
            self.loss = nn.BCELoss()
        if self.criterion != None:
            self.loss = self.criterion
        return self.loss


class TrainNN():
    def __init__(self, model, x, y, learning_rate=0.0001, dateloader_rate=1, gradient_threshold=0.0001, n_epochs=10000):
        self.model= model
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.lr = learning_rate
        self.dateloader_rate = dateloader_rate
        self.gradient_threshold = gradient_threshold
        self.n_epochs = n_epochs

    def get_dataloader(self):
        dataset_ = TensorDataset(self.x, self.y)
        dl = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        return dl
    
    def nn_training(self, loss_func=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # n_epochs = 10000
        epoch_losses = []
        dl_ = self.get_dataloader()

        if loss_func == None:
            lf = LossFunction()
            loss_criterion = lf.loss_criterion()
        else:
            lf = loss_func
            loss_criterion = loss_func.loss_criterion()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            gradient_norms = []
            total_losses = []
            for batch_x, batch_y in dl_:
                optimizer.zero_grad()
                predictions = self.model.forward(batch_x)
                loss = loss_criterion(predictions, batch_y.view(-1,1))
                if lf.l1_lambda != None and lf.binary_lambda != None:
                    penalty_loss = self.model.l1_n_binary_penalty(lf.l1_lambda, lf.binary_lambda)
                elif lf.l1_lambda != None:
                    penalty_loss = self.model.l1_penalty(lf.l1_lambda)
                elif lf.binary_lambda != None :
                    penalty_loss = self.model.binary_penalty(lf.binary_lambda)
                else:
                    penalty_loss = 0
                # print(penalty_loss)
                total_loss = loss + penalty_loss
                total_losses.append(total_loss.item())
                total_loss.backward()  
                optimizer.step()
                
                epoch_loss += total_loss.item()

            epoch_losses.append(np.mean(total_losses))

            for param in self.model.parameters():
                gradient_norms.append(torch.norm(param.grad).item())

            if all(norm < self.gradient_threshold for norm in gradient_norms):
                print(f'Early stopping at Epoch {epoch + 1} due to small gradients.')
                break
            if epoch+1 == self.n_epochs:
                print(f'End of epoch number')

        return self.model, epoch_losses


class BinaryTrainNN():
    def __init__(self, model, x, y, learning_rate=0.0001, dateloader_rate=1, gradient_threshold=0.0001, n_epochs=10000):
        self.model= model
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.lr = learning_rate
        self.dateloader_rate = dateloader_rate
        self.gradient_threshold = gradient_threshold
        self.n_epochs = n_epochs

    def get_dataloader(self):
        dataset_ = TensorDataset(self.x, self.y)
        dl = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        return dl

    def nn_training(self, loss_func=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # n_epochs = 10000
        epoch_losses = []
        dl_ = self.get_dataloader()
        # dataset_ = TensorDataset(self.x, self.y)
        # dl_ = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        if loss_func == None:
            lf = LossFunction()
            loss_criterion = lf.loss_criterion()
        else:
            lf = loss_func
            loss_criterion = loss_func.loss_criterion()

        for epoch in range(self.n_epochs):
            # print(epoch)
            epoch_loss = 0.0
            gradient_norms = []
            total_losses = []
            for batch_x, batch_y in dl_:
                optimizer.zero_grad()

                if lf.l1_lambda != None and lf.binary_lambda != None:
                    penalty_loss = self.model.l1_n_binary_penalty(lf.l1_lambda, lf.binary_lambda)
                elif lf.l1_lambda != None:
                    penalty_loss = self.model.l1_penalty(lf.l1_lambda)
                elif lf.binary_lambda != None :
                    penalty_loss = self.model.binary_penalty(lf.binary_lambda)
                else:
                    penalty_loss = 0

                for name, param in self.model.named_parameters():
                    if name == 'hidden2.weight':
                        unbinarized_weights = param.detach()
                        binarized_weights = torch.where(param >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                        param.data = binarized_weights

                predictions = self.model.forward(batch_x)  
                loss = loss_criterion(predictions, batch_y.view(-1,1)) 
                total_loss = loss + penalty_loss
                total_losses.append(total_loss.detach().numpy())
                total_loss.backward()

                for name, param in self.model.named_parameters():
                    if name == 'hidden2.weight': 
                        if param.grad is not None:
                            param.data = unbinarized_weights
                        else:
                            print('grad is none')

                optimizer.step()
                epoch_loss += total_loss.item()

            epoch_losses.append(np.mean(total_losses))

            for param in self.model.parameters():
                gradient_norms.append(torch.norm(param.grad).item())

            if all(norm < self.gradient_threshold for norm in gradient_norms):
                print(f'Early stopping at Epoch {epoch + 1} due to small gradients.')
                break
            
            if epoch + 1 == self.n_epochs:
                print('End of Number of Epochs')
        

        return self.model, epoch_losses