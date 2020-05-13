import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch import optim
from torch import nn
from collections import deque
import pandas as pd
import numpy as np
import random
import warnings 
warnings.simplefilter('ignore')

# model
class critic(nn.Module):
    ''' 
    input : state
    output : state_value(scalar)
    '''
    def __init__(self, input_dim):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(input_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)

    def forward(self,state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        
        # Linear output
        return self.fc3(state)

class actor(nn.Module):
    '''
    input : state 
    output : action_probs(vector)
    '''
    def __init__(self, input_dim, output_dim):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(input_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,output_dim)
    
    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = F.relu(self.fc2(logits))
        
        # softmax output
        return F.softmax(self.fc3(logits),dim=1)

# agent
class PPOAgent:
    def __init__(self,env,gamma = 0.99,
                 clip = 0.2,
                 lr = 1e-3,
                 K_epoch = 4,
                 lam = 0.95):
        
        # common
        self.device = "cuda"
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Hyperparamters
        self.gamma = gamma
        self.clip = clip
        self.lr = lr
        self.K_epoch = int(K_epoch)
        self.lam = lam
        
        # critic
        self.critic = critic(self.obs_dim).to(self.device)
        self.critic.apply(self._weights_init)
        
        # actor_new
        self.actor_new = actor(self.obs_dim,self.action_dim).to(self.device)
        self.actor_new.apply(self._weights_init)
        
        # actor_old,sync
        self.actor_old = actor(self.obs_dim,self.action_dim).to(self.device)
        self.sync()
        
        # optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)
        self.actor_optimizer = optim.Adam(self.actor_new.parameters(),lr=lr)
        
        # recorder
        self.recorder = {'a_loss':[],
                         'v_loss':[],
                         'e_loss':[],
                         'ratio':[]}
            
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)
    
    def sync(self):
        for old_param, new_param in zip(self.actor_old.parameters(),self.actor_new.parameters()):
            old_param.data.copy_(new_param.data)
    
    def get_action(self,state):
        state = torch.FloatTensor([state]).to(self.device)
        probs = self.actor_new(state) # softmax_probs
        dist = Categorical(probs) # Categorical distribution
        act = dist.sample() # smaple action from this Categorical distribution
        return act.detach().item()
    
    def get_value(self,state):
        state = torch.FloatTensor([state]).to(self.device)
        value = self.critic(state)
        return value.item()
    
    def compute_returns(self,rewards):
        returns = []
        G = 0
        
        for r in rewards[::-1]:
            G = r + self.gamma*G
            returns.insert(0,G)
        
        returns = np.array([i for i in returns]).ravel()
        
        return torch.FloatTensor(returns).to(self.device).view(-1, 1)
    
    def compute_gae(self,next_values, rewards, masks, values):
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):    
            delta = rewards[step] + self.gamma * next_values[step] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        
        returns = np.array([i.detach().item() for i in returns]).ravel()
        
        return torch.FloatTensor(returns).to(self.device).view(-1, 1)
    
    def normalize(self,x):
        return (x - x.mean()) / x.std()
    
    def update(self,trajectory):
        
        # get trajectory
        state = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        action = torch.LongTensor([sars[1] for sars in trajectory]).to(self.device).view(-1, 1)
        rewards = [sars[2] for sars in trajectory]
        next_state = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        done = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device).view(-1, 1)
        
        # update K_epoch
        for _ in range(self.K_epoch):    
            
            # calculate critic loss
            values = self.critic(state)
            returns = self.normalize(self.compute_returns(rewards))
            advantage = returns - values
            critic_loss = 0.5 * (advantage**2).mean()
            self.recorder['v_loss'].append(critic_loss.item())
            
            # calculate actor_loss
            new_p = torch.gather(self.actor_new(state),1,action)
            old_p = torch.gather(self.actor_old(state),1,action)
            ratio = new_p / old_p
            self.recorder['ratio'].append(ratio.mean().item())
            
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio,1 - self.clip,1 + self.clip) * advantage.detach()
            entropy_loss = Categorical(self.actor_new(state)).entropy().mean()
            self.recorder['e_loss'].append(entropy_loss.item())
            
            actor_loss = -torch.min(surr1,surr2).mean() - 0.001 * entropy_loss
            self.recorder['a_loss'].append(actor_loss.item())
            
            # update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # sync actor_old and actor_new
        self.sync()
    
    def train(self,max_episodes,max_steps,batch_size,render_area,score_area,progress_bar):
        episode_rewards = []
        for episode in range(max_episodes):
            
            # initialize new game
            state = self.env.reset()
            trajectory = [] # [[s, a, r, s', done], [], ...]
            episode_reward = 0
            done = False
            
            # game loop
            while not done:
                render_area.image(self.env.render(mode='rgb_array')) #render
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append([state, action, reward, next_state, done])
                episode_reward += reward
                state = next_state
            
            # game over
            print("Episode " + str(episode) + ": " + str(episode_reward))
            episode_rewards.append(episode_reward)
            self.update(trajectory)
            
            # update score_area and progressbar
            row = pd.DataFrame([[episode_reward,np.mean(episode_rewards[-10:])]],columns=['reward','rolling_reward']).astype("float")
            score_area.add_rows(row)
            progress_bar.progress((episode + 1)/max_episodes)
        
        return episode_rewards
    
    def play(self,max_episodes):
        
        for episode in range(max_episodes):
            
            # initialize new game
            state = env.reset()
            episode_reward = 0
            done = False
            
            # game loop
            while not done:
                self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            
            # game over
            print("Episode " + str(episode) + ": " + str(episode_reward))
        
        self.env.close()