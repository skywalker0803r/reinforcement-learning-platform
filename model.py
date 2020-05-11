import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F 

class DQN(nn.Module):
    '''
    
    input:state
    output:qvals
    
    '''
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.Linear1 = nn.Linear(input_dim,128) 
        self.Linear2 = nn.Linear(128,128)
        self.Linear3 = nn.Linear(128,output_dim)

    def forward(self, state):
        
        state = F.relu(self.Linear1(state))
        state = F.relu(self.Linear2(state))
        qvals = self.Linear3(state)
        
        return qvals

class ActorCritic(nn.Module):
    '''
    
    input:state
    output1:action_probs
    output2:state_value
    
    '''
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Linear(input_dim,128)
        
        self.policy1 = nn.Linear(128,128) 
        self.policy2 = nn.Linear(128,output_dim)
        
        self.value1 = nn.Linear(128,128)
        self.value2 = nn.Linear(128,1)
        
    def forward(self, state):
        state = F.relu(self.common(state))
        
        logits = F.relu(self.policy1(state))
        logits = self.policy2(logits)
        
        value = F.relu(self.value1(state))
        value = self.value2(value)
        
        return F.softmax(logits),value

class Critic(nn.Module):
    '''
    
    input:state,action(s,a)
    output:state_value(v)
    C(s,a) = v
    '''
    def __init__(self,obs_dim,action_dim):
        super(Critic,self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim,128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,1)

    def forward(self,obs,action):
        value = F.relu(self.linear1(torch.cat((obs,action),dim=1)))
        value = F.relu(self.linear2(value))
        value = self.linear3(value)
        return value

class Actor(nn.Module):
    '''
    
    input:state
    output:action_probs
    
    '''
    def __init__(self, obs_dim, action_dim,ddpg=False):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(obs_dim,128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,action_dim)
        self.ddpg = ddpg

    def forward(self,obs):
        logits = F.relu(self.linear1(obs))
        logits = F.relu(self.linear2(logits))
        logits = self.linear3(logits)
        if self.ddpg == True:
            return logits
        if self.ddpg == False:
            return F.softmax(logits)