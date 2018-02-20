import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal

class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()
        

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False,missions=False):
        if missions is not False :
            value, x, states = self(inputs, states, masks,missions)
        else:
            value, x, states = self(inputs, states, masks)
            
            
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions,missions=False):
        if missions is not False :
            value, x, states = self(inputs, states, masks,missions)
        else:
            value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class RecMLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(RecMLPPolicy, self).__init__()

        self.action_space = action_space
        assert action_space.__class__.__name__ == "Discrete"
        num_outputs = action_space.n

        self.p_fc1 = nn.Linear(num_inputs, 64)
        self.p_fc2 = nn.Linear(64, 64)
        
         # models used to mix text and image inputs
        self.adapt_1 = nn.Linear(4096,num_inputs)
        self.adapt_2 = nn.Linear(num_inputs,64)
        self.adapt_3 = nn.Linear(2*64,64)



        self.v_fc1 = nn.Linear(64, 64)
        self.v_fc2 = nn.Linear(64, 32)
        self.v_fc3 = nn.Linear(32, 1)
        
        self.a_fc1 = nn.Linear(64, 64)
        self.a_fc2 = nn.Linear(64, 64)
        #self.a_fc3 = nn.Linear(32, action_space.n)
        
       

        # Input size, hidden size
        self.gru = nn.GRUCell(64, 64)

        self.dist = Categorical(64, num_outputs)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        """
        Size of the recurrent state of the model (propagated between steps
        """
        return 64

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        orthogonal(self.gru.weight_ih.data)
        orthogonal(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks, missions=False):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)

        x = self.p_fc1(inputs)
        x = F.tanh(x)
        x = self.p_fc2(x)
        x = F.tanh(x)
        
        if missions is not False:
            missions=self.adapt_1(missions)
            missions=self.adapt_2(missions)
            missions=torch.cat([missions, x],dim=1)
            x= self.adapt_3(missions)

        

        assert inputs.size(0) == states.size(0)
        
        x = states = self.gru(x, states * masks)
        
        actions = self.a_fc1(x)
        actions = F.tanh(actions)
        actions = self.a_fc2(actions)
        #actions = F.tanh(actions)
        #actions = self.v_fc3(actions)

        value = self.v_fc1(x)
        value = F.tanh(value)
        value = self.v_fc2(value)
        value = F.tanh(value)
        value = self.v_fc3(value)

        return value, actions, states

class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)
        
        self.adapt_1 = nn.Linear(4096,num_inputs)
        self.adapt_2 = nn.Linear(2*num_inputs,num_inputs)
        

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks,missions=False):
        batch_numel = reduce(operator.mul, inputs.size()[1:], 1)
        inputs = inputs.view(-1, batch_numel)
        
        
        if missions is not False:
            missions=self.adapt_1(missions)
            missions=torch.cat([missions, inputs],dim=1)
            inputs=self.adapt_2(missions)
            

        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x, states

def weights_init_cnn(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, use_gru):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        if use_gru:
            self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init_cnn)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        if hasattr(self, 'gru'):
            x = states = self.gru(x, states * masks)

        return self.critic_linear(x), x, states
