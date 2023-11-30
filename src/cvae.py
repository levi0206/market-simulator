import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import utils

class CVAE(nn.Module):
    def __init__(self, data, data_cond, latent_dim, hidden_dim=50, alpha=0.2):
        super(CVAE,self).__init__()
        if not torch.is_tensor(data):
            self.data = torch.tensor(data,dtype=torch.float32).squeeze()
        else:
            self.data = data.squeeze()
        if not torch.is_tensor(data_cond):
            self.data_cond = torch.tensor(data_cond,dtype=torch.float32)
        else:
            self.data_cond = data_cond

        if len(data_cond.shape)>2:
            self.data_cond = self.data_cond.squeeze()

        # Check input shape
        assert len(self.data.shape)==2, "Shape of input data tensor should be 2"
        assert len(self.data_cond.shape)==2, "Shape of input condition tensor should be 2"

        print("Data shape:{}".format(self.data.shape))
        print("Data condition shape:{}".format(self.data_cond.shape))
    
        assert self.data.max() <= 1. and self.data.min() >=0., \
            "All features of the dataset must be between 0 and 1."
        assert self.data_cond.max() <= 1. and self.data_cond.min() >=0., \
            "All features of the dataset must be between 0 and 1."

        self.input_dim = self.data.shape[1]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        self.encoder = nn.Sequential(
            nn.Linear(self.data.shape[1]+self.data_cond.shape[1],hidden_dim*2),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.LeakyReLU(0.3),
        )

        self.encode_fc = nn.Linear(hidden_dim,latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.data_cond.shape[1], hidden_dim*2),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim*2, self.hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

        self.MODEL_NAME = 'model.pth'

    def encode(self,x):
        x = torch.cat((x, self.data_cond), dim=1)  
        x = x.flatten(start_dim=1)  
        # print("concatenated tensor shape:{}".format(x.shape))
        x = self.encoder(x)  
        mu = nn.LeakyReLU(0.3)(self.encode_fc(x))  
        sigma = nn.LeakyReLU(0.3)(self.encode_fc(x)) 

        epsilon = torch.randn(mu.shape)
        z = mu + torch.mul(epsilon, torch.exp(sigma/2.)) 

        return z, mu, sigma
    
    def decode(self,z,cond):
        z = torch.cat([z, cond], dim=1)  
        reconstruct = self.decoder(z)             
        reconstruct = reconstruct.view(-1,self.input_dim)  

        return reconstruct
    
    def forward(self,x):
        latent, mu, sigma = self.encode(x)
        # latent = torch.cat([latent, self.data_cond], dim=1) 
        reconstruct = self.decode(latent,self.data_cond)
        return reconstruct

    def customloss(self,x,mu,logsigma):
        reconstruct_loss = F.mse_loss(x, self.data,reduction='sum')
        KLD = - 0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        loss = (1-self.alpha)*reconstruct_loss + self.alpha*KLD
        
        return loss
    
    def train(self, n_epochs=10000, learning_rate=0.005):
        loss_record = []

        # Early stop
        stop = 0
        def is_stop(stop):
            if stop > 1000:
                return True
            else:
                return False
            
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            latent, mu, logsigma = self.encode(self.data)
            reconstruct = self.decode(latent,self.data_cond)
            loss = self.customloss(reconstruct,mu,logsigma)
            loss_record.append(loss.item())
            loss.backward()
            optimizer.step()

            if loss.item()==min(loss_record):
                print("Epoch {}: {:.4f}".format(epoch+1,loss.item()))
                print("saving model with loss {:.4f}".format(loss.item()))
                torch.save(self.state_dict(), "%s" % self.MODEL_NAME)
                stop = 0

            if is_stop(stop):
                print("Early stop at {}".format(epoch+1))
                break
            else:
                stop += 1

    def generate(self, cond, n_samples=None):

        self.load_state_dict(torch.load(self.MODEL_NAME))
        
        cond = utils.as_float_array(cond)

        if n_samples is not None:
            randoms = torch.rand(n_samples,self.latent_dim)
            cond = [list(cond)] * n_samples
            cond = torch.tensor(cond,dtype=torch.float32)
            cond = cond.view(n_samples,self.data_cond.shape[1])
        else:
            randoms = torch.rand(1,self.latent_dim)
            cond = [list(cond)] 
            cond = torch.tensor(cond,dtype=torch.float32)
            cond = cond.view(1,self.data_cond.shape[1])

        with torch.no_grad():
            z = torch.cat((randoms, cond), dim=1)
            generated_samples = self.decoder(z)

        if n_samples is None:
            return generated_samples[0]

        return generated_samples