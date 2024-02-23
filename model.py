import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, bias_conv =True, bias_fc=True):
        super(Net, self).__init__()
        # r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out:26, j_out:1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=bias_conv)
        # r_in:3, n_in:26, j_in:1, s:1, r_out:5, n_out:24, j_out:1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=bias_conv)
        # Applied Max pooling with a stride of 2
        # r_in:5, n_in:24, j_in:1, s:2, r_out:6, n_out:12, j_out:2

        # r_in:6, n_in:12, j_in:2, s:1, r_out:10, n_out:10, j_out:2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=bias_conv)
        # r_in:10, n_in:10, j_in:2, s:1, r_out:14, n_out:8, j_out:2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=bias_conv)
        # Applied Max pooling with a stride of 2
        # r_in:14, n_in:8, j_in:2, s:2, r_out:16, n_out:4, j_out:2
        
        self.fc1 = nn.Linear(256*4*4, 50, bias=bias_fc) #defines the first fully connected (linear) layer (fc1). It takes an input size of 256*4*4 (which is the result of flattening the output from the convolutional layers), outputs 50 neurons
        self.fc2 = nn.Linear(50, 10, bias=bias_fc)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # applies the ReLU activation function
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # applies the softmax function along the specified dimension (dim=1)