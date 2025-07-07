import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import scipy as sp

# setup gpu / npu
torch.set_default_dtype(torch.float32)
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"

# pokemon emerald rng
rng_seed = 0 
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

lap_1d = [[[-1/12, 4/3, -2.5, 4/3, -1/12]]] # 2nd derivative approximation kernel

# trained upscaler model
class Upscaler(nn.Module):

    def __init__(self):

        super(Upscaler, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(1, 8, kernel_size=9, padding=5 // 2, stride=3, output_padding=2, bias=True),
            nn.Sigmoid(),
            nn.ConvTranspose1d(8, 8, kernel_size=9, padding=5 // 2, stride=3, output_padding=2, bias=True),
            nn.Conv1d(8, 1, 1, 1, padding=0, bias=True)
        )

    def forward(self, input):
        return self.net(input)
    
# recurrent convolutional cell
class RCNNCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, input_kernel_size):

        super().__init__()

        # hyperparameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = 5
        self.input_stride = 1
        self.dx = 0.01
        self.dt = 0.5
        self.mu_up = 3.89e-5 # TODO: this is a constant that i do not understand, but without it i get exploding gradients

        # laplace parameters?
        # TODO: WHAT DOES THIS MEAN WHAT DOES THIS DO I DONT GET THIS
        self.CA = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)
        self.CB = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)

        # laplace convolution
        self.laplace = nn.Conv1d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.laplace.weight.data = 1/self.dx**2*torch.tensor(lap_1d, dtype=torch.float32)
        self.laplace.weight.requires_grad = False

        # Nonlinear term for u (up to 3rd order)?
        # TODO: WHAT DOES THIS MEAN WHAT DOES THIS DO
        self.Wh1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4 = nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        
        self.filter_list = [self.Wh1, self.Wh2, self.Wh3, self.Wh4]
        self.init_filter(self.filter_list, c=0.02)

    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filters for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)

    def forward(self, h):

        # periodic padding
        h_pad = torch.cat((    h[:, :, -2:],     h,     h[:, :, 0:2]), dim=2)
        u_pad = h_pad[:, 0:1, ...]  # 1xTx37
        u_prev = h[:, 0:1, ...]     # 1xTx33

        u_res = self.mu_up*torch.sigmoid(self.CA)*self.laplace(u_pad) + self.Wh4( self.Wh1(h)*self.Wh2(h)*self.Wh3(h) )
        u_next = u_prev + u_res * self.dt
        ch = u_next

        return ch, ch # #TODO: WHY AM I RETURNING TWICE


class RCNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, init_state_low, input_kernel_size,
                       step=1, effective_step=[1]):
        
        super(RCNN, self).__init__()

        # hyperparameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = 1  # always 1. idk why we do that...
        self.input_kernel_size = input_kernel_size
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.init_state_low = init_state_low.unsqueeze(0).unsqueeze(0)
        self.init_state = []

        # upscaler for initial state
        self.UpconvBlock = Upscaler()

        # setup CRNN cell
        name = 'crnn_cell' # what?
        cell = RCNNCell(
            input_channels = self.input_channels,
            hidden_channels = self.hidden_channels,
            input_kernel_size = self.input_kernel_size
        )
        setattr(self, name, cell)
        self._all_layers.append(cell)

    def forward(self):

        # initial setup
        self.init_state = self.UpconvBlock(self.init_state_low) # upscale initial state

        internal_state = []
        outputs = [self.init_state]
        second_last_state = []

        # iterate over time through recurrent nn shenanigans
        for step in range(self.step):
            name = 'crnn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_state
                internal_state = h

            h = internal_state
            # hidden state and output
            h, o = getattr(self, name)(h)
            internal_state = h

            if step == (self.step - 2):
                #  last output is a dummy for central FD
                second_last_state = internal_state.clone()

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs, second_last_state
    

class Conv1dDerivative(nn.Module):

    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        super(Conv1dDerivative, self).__init__()
        self.deno = deno
        self.kernel_size = kernel_size
        self.input_channels = 1
        self.output_channels = 1
        self.name = name

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        # Fixed gradient operator

        # set weights to passed in filter (probably laplace)
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)


    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno
    

class Conv1dDerivative(nn.Module):

    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.deno = deno  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno
    

class LossGenerator(nn.Module):

    def __init__(self, dt = (1.0/2), dx = (1.0/100)):
        self.dt = dt
        self.dx = dx
       
        super(LossGenerator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv1dDerivative(
            DerFilter = lap_1d,
            deno = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').to(device)

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 1, 0]]],
            deno = (dt*1),
            kernel_size = 3,
            name = 'partial_t').to(device)
        
    def get_phy_loss(self, output):

        # du2/d2x
        laplace_u = self.laplace(output[0:1, :, :]) # 1xT-2x32 (1x999x32)

        u = output[0:1, :, 1:-1]
        len_t = u.shape[1]
        len_x = u.shape[2]

        # u_conv1d = u.permute(2, 0, 1) # TODO: i don't really understand why... [x, u, step]
        # du_dt = self.dt(u_conv1d) # length is 2 smaller since no padding
        # du_dt = du_dt.permute(1, 2, 0) # undo initial permutation

        du_dt = self.dt(u)

        u = output[0:1, :, 2:-2]

        # make sure dimensions good
        assert laplace_u.shape == du_dt.shape
        assert du_dt.shape == u.shape

        # heat equation
        alpha = 0.1
        heat_eq_rhs = alpha*laplace_u
        residual = du_dt - heat_eq_rhs

        return residual

def get_ic_loss(model):

    init_state_upscaled = F.interpolate(model.init_state_low, (151), mode='linear')

    init_state_prediction = model.UpconvBlock(model.init_state_low)

    loss_ic = nn.MSELoss()(init_state_prediction, init_state_upscaled) #TODO: SHOULDNT THIS NOT BE USING THE UPSCALER????

    return loss_ic

def loss_gen(output, loss_fxn):
    
    # calculate PDE loss
    phys_loss = loss_fxn.get_phy_loss(output)
    loss = nn.MSELoss()(phys_loss, torch.zeros_like(phys_loss).to(device))

    return loss
    
def pretrain_upscaler(Upconv, init_state_low, epochs=4000):
    init_state_upscaled = F.interpolate(init_state_low.unsqueeze(0).unsqueeze(0), (151), mode='linear')

    optimizer = optim.Adam(Upconv.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.99)

    # for epoch in range(epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        init_state_pred = Upconv(init_state_low.unsqueeze(0))
        loss = nn.MSELoss()(init_state_pred, init_state_upscaled)
        loss.backward(retain_graph=True)
        print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
        optimizer.step()
        scheduler.step()

def train(model, truth, epochs, time_batch_size, lr, dt, dx, isContinuing):

    # upscale truth
    upscaled_t_steps = []
    for i in range(truth.shape[1]):
        t_step = truth[:,i:i+1,:]
        t_step_upscaled = model.UpconvBlock(t_step)
        upscaled_t_steps.append(t_step_upscaled)

    upscaled_truth = torch.stack(upscaled_t_steps).permute(2,1,0,3)[0]

    train_loss_list = []

    if isContinuing:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.985)

    loss_fxn = LossGenerator(dt, dx)

    for epoch in range(epochs):

        # upscale truth using pretrained upscaler
        # I feel like this shouldn't be in the loop, but I get the most indecipherable error
        # I've ever seen if it isn't. THe error message is phrased like a RIDDLE
        upscaled_t_steps = []
        for i in range(truth.shape[1]):
            t_step = truth[:,i:i+1,:]
            t_step_upscaled = model.UpconvBlock(t_step)
            upscaled_t_steps.append(t_step_upscaled)
        upscaled_truth = torch.stack(upscaled_t_steps).permute(2,1,0,3)[0]

        optimizer.zero_grad()

        num_time_batch = 1

        batch_loss, phy_loss, ic_loss, data_loss, val_loss = [0]*5

        output, second_last_state = model()

        output = torch.cat(tuple(output), dim=0)

        pred = output.permute(1,0,2)[:,:-1,:] # 1xTxX
        gt = upscaled_truth.to(device)

        # # split into training set and validation set
        idx = int(pred.shape[1]*0.8)
        pred_tra = pred[:idx]
        pred_val = pred[idx:]
        gt_tra =   gt[:idx]
        gt_val = gt[idx:]

        # # clamp to avoid NaN
        # pred_tra = torch.clamp(pred_tra, -1e3, 1e3)
        # gt_tra = torch.clamp(gt_tra, -1e3, 1e3)

        # pred = torch.clamp(pred, -1e3, 1e3)
        # gt = torch.clamp(pred, -1e3, 1e3)

        # compute losses
        loss_data = nn.MSELoss(reduction='sum')(pred, gt)
        loss_valid = nn.MSELoss(reduction='sum')(pred_val, gt_val)
        loss_ic  = get_ic_loss(model)
        loss_phy = loss_gen(output, loss_fxn)

        # weight losses (physics loss only used for validation)
        loss = loss_data + loss_ic
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)

        batch_loss += loss.item()

        # unpack from tensors for printing
        phy_loss = loss_phy.item()
        ic_loss = loss_ic.item()
        data_loss = loss_data.item()
        val_loss = loss_valid.item()

        optimizer.step()
        scheduler.step()

        # print into
        print('[%d/%d %d%%] loss: %.7f, ic_loss: %.7f, data_loss: %.7f, val_loss: %.7f, phy_loss: %.8f' % ((epoch+1), epochs, ((epoch+1)/epochs*100.0),
              batch_loss, ic_loss, data_loss, val_loss, phy_loss))
        
        train_loss_list.append(batch_loss)

        # save model every 100 epochs
        if (epoch+1)%100 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model!!!')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './PeRCNN(Paper)/1d_heat/model/checkpoint.pt')

    return train_loss_list

def save_model(model, model_name, save_path):
    torch.save(model.state_dict(), save_path + model_name + '.pt')

def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./PeRCNN(Paper)/1d_heat/model/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.98)
    return model, optimizer, scheduler


if __name__ == '__main__':
    ################# prepare the input dataset ####################
    time_steps = 800   # 200->400->800 multi-stage training works best, then 2500 for inference.
    dt = 0.5
    dx = 1.0/100
    dy = 1.0/100

    ################### define the Initial conditions ####################
    data = sp.io.loadmat('./PeRCNN(Paper)/1d_heat/1x1001x15_heat_eq_data.mat')['tensor']
    datamat = torch.from_numpy(np.transpose(data, (0, 1, 2)).astype(np.float32)) # 1x1001x15
    truth_clean = datamat[:,:1001]  # 1x1001x15
    # Add noise 10%
    # UV = add_noise(torch.tensor(datamat), pec=0.1) #TODO
    # Retrieve initial condition
    IC = datamat[:, 0:1, :] # 1x1x15 IC

    truth = datamat[:,:time_steps+1]

    ################# build the model #####################
    # define the mdel hyperparameters
    time_batch_size = time_steps
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    n_iters = 10000   # 10000 for 200 steps, 5000 for 4000 steps, 5000 for 800 steps
    learning_rate = 1e-3
    save_path = './PeRCNN(Paper)/1d_heat/model/'

    # Low-res initial condition
    U0_low = IC[0, 0, :]
    init_state_low = torch.tensor((U0_low)).to(device)

    model = RCNN(
        input_channels = 2, 
        hidden_channels = 8,
        init_state_low = init_state_low,
        input_kernel_size = 5,
        step = steps, 
        effective_step = effective_step
    ).to(device)

    # train the model
    start = time.time()
    cont = True   # if continue training (or use pretrained model), set cont=True
    if not cont:
        pretrain_upscaler(model.UpconvBlock, init_state_low, epochs=5000)
    train_loss_list = train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, isContinuing=cont)

    print('The training time is: ', (time.time()-start))

    # Do the forward inference
    output, _ = model()
    output = torch.cat(tuple(output), dim=0) # concatenate the list of states from the model into one tensor

