import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd


class GeneralRegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(GeneralRegressionNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2, bias=True)
        self.hidden3 = nn.Linear(hidden_size2, hidden_size3, bias=False)
        self.output = nn.Linear(hidden_size3, output_size, bias=False)

   
    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.hidden3(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        # x = torch.sigmoid(x)
        return x

    
    def predict(self, input):
        self.eval()
        input = torch.tensor(input, dtype=torch.float32)

        with torch.no_grad():
            output = self.forward(input)

        self.train()
        
        return output.numpy()
    
    def binary_loss(self):
        binary_loss = 0.0
        w_ = []
        for name, param in self.named_parameters():
            if name == 'hidden3.weight':
                w_.append(param.reshape(-1,1))
            
        for v in w_:
            for v_i in v:
                if torch.abs(v_i) < torch.abs(v_i-1):
                    b = 0
                else:
                    b = 1
                binary_loss += torch.sum(torch.tensor(b))
        return binary_loss
    
    def binarise_model(self, layer=3):
        layer_dict = {1: 'hidden1.weight', 2: 'hidden2.weight', 3:'hidden3.weight', 4:'output.weight'}
        for name, param in self.named_parameters():
            if name == layer_dict[layer]:
                param.data = torch.where(param.data >= 0.5, torch.tensor(1.0), torch.tensor(0.0))


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


class TrainNN():
    def __init__(self, x, y, learning_rate=0.0001, dateloader_rate=1):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.lr = learning_rate
        self.dateloader_rate = dateloader_rate

    def get_dataloader(self):
        dataset_ = TensorDataset(self.x, self.y)
        dl = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        return dl

    def nn_training(self, model, loss_criterion = nn.MSELoss(), gradient_threshold=0.0001):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        n_epochs = 20000
        epoch_losses = []
        dl_ = self.get_dataloader()
        # dataset_ = TensorDataset(self.x, self.y)
        # dl_ = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            gradient_norms = []
            total_losses = []
            for batch_x, batch_y in dl_:
                optimizer.zero_grad()
                predictions = model.forward(batch_x)  
                loss = loss_criterion(predictions, batch_y.view(-1,1)) 
                total_loss = loss 
                total_losses.append(total_loss.detach().numpy())
                total_loss.backward()  
                optimizer.step()
                epoch_loss += total_loss.item()

            epoch_losses.append(np.mean(total_losses))

            for param in model.parameters():
                gradient_norms.append(torch.norm(param.grad).item())

            if all(norm < gradient_threshold for norm in gradient_norms):
                print(f'Early stopping at Epoch {epoch + 1} due to small gradients.')
                break
        

        return model, epoch_losses


class BinaryTrainNN():
    def __init__(self, x, y, learning_rate=0.0001, dateloader_rate=1):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.lr = learning_rate
        self.dateloader_rate = dateloader_rate

    def get_dataloader(self):
        dataset_ = TensorDataset(self.x, self.y)
        dl = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        return dl

    def nn_training(self, model, loss_criterion = nn.MSELoss(), gradient_threshold=0.0001):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        n_epochs = 30000
        epoch_losses = []
        dl_ = self.get_dataloader()
        # dataset_ = TensorDataset(self.x, self.y)
        # dl_ = DataLoader(dataset=dataset_, batch_size=int(len(self.x)*self.dateloader_rate), shuffle=True)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            gradient_norms = []
            total_losses = []
            for batch_x, batch_y in dl_:
                optimizer.zero_grad()

                for name, param in model.named_parameters():
                    if name == 'hidden3.weight':
                        unbinarized_weights = param.detach()
                        binarized_weights = torch.where(param >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                        param.data = binarized_weights

                predictions = model.forward(batch_x)  
                loss = loss_criterion(predictions, batch_y.view(-1,1)) 
                total_loss = loss 
                total_losses.append(total_loss.detach().numpy())
                total_loss.backward()

                for name, param in model.named_parameters():
                    if name == 'hidden3.weight': 
                        if param.grad is not None:
                            param.data = unbinarized_weights
                        else:
                            print('grad is none')

                optimizer.step()
                epoch_loss += total_loss.item()

            epoch_losses.append(np.mean(total_losses))

            for param in model.parameters():
                gradient_norms.append(torch.norm(param.grad).item())

            if all(norm < gradient_threshold for norm in gradient_norms):
                print(f'Early stopping at Epoch {epoch + 1} due to small gradients.')
                break
        

        return model, epoch_losses