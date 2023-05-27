import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F 
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from utils import *
from model import *

'''
this file contain all agent like DQNAgent,A2CAgent,DDPGAgent....
all agent have get_action and update method and default params
'''
    
class DQNAgent:

    def __init__(self, env,learning_rate=1e-3,gamma=0.99,eps=0.05,buffer_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        
        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.replay_buffer = BasicBuffer(max_size=int(buffer_size))
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.05):
        state = torch.FloatTensor([state]).to(self.device)
        qvals = self.model.forward(state)
        action = qvals.cpu().detach().numpy().argmax()
        
        if(np.random.randn() < self.eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).squeeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1,actions).squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q,1)[0]
        expected_Q = rewards + self.gamma * max_next_Q * (1-dones)

        loss = self.MSE_loss(curr_Q, expected_Q)
        
        return loss

    def update(self, batch_size):
    	if len(self.replay_buffer) > batch_size:
            batch = self.replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self,max_episodes,batch_size,render_area,score_area,progress_bar):
        episode_rewards = []
        for episode in range(max_episodes):
            # init all
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            # not done
            while not done:
                render_area.image(self.env.render(mode='rgb_array'))
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward 
                self.update(batch_size)
                state = next_state   
            
            # is done
            episode_rewards.append(episode_reward)
            print("Episode:{} reward:{}".format(episode,episode_reward))      
            row = pd.DataFrame([[episode_reward,np.mean(episode_rewards[-10:])]],
            	columns=['reward','rolling_reward']).astype("float")
            score_area.add_rows(row)
            progress_bar.progress((episode + 1)/max_episodes)            
        
        return episode_rewards

class A2CAgent():

    def __init__(self, env, gamma=0.99, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.gamma = gamma
        self.lr = lr
        
        self.model = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits, _ = self.model.forward(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()
    
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)
        
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor(
        	[self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device)
             * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        logits, values = self.model.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_loss = F.mse_loss(values, value_targets.detach())
        
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        
        total_loss = policy_loss + value_loss - 0.001 * entropy 
        return total_loss
        
    def update(self, trajectory):
        loss = self.compute_loss(trajectory)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self,max_episodes,batch_size,render_area,score_area,progress_bar):
        episode_rewards = []
        for episode in range(max_episodes):
            # init all
            state = self.env.reset()
            episode_reward = 0
            done = False
            trajectory = []
        
            # not done
            while not done:
                render_area.image(self.env.render(mode='rgb_array'))
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append([state, action, reward, next_state, done])
                episode_reward += reward 
                state = next_state   
	        
            # is done
            self.update(trajectory)
            episode_rewards.append(episode_reward)
            print("Episode:{} reward:{}".format(episode,episode_reward))      
            row = pd.DataFrame([[episode_reward,np.mean(episode_rewards[-10:])]],
            	columns=['reward','rolling_reward']).astype("float")
            score_area.add_rows(row)
            progress_bar.progress((episode + 1)/max_episodes)            
        
        return episode_rewards

#=================================================================================================================================================
class DDPGAgent:
    
    def __init__(self, env, gamma=0.99, tau=1e-2, buffer_maxlen=100000, critic_learning_rate=1e-3, actor_learning_rate=1e-3):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # env
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim,ddpg=True).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim,ddpg=True).to(self.device)
        
        # Copy actor target parameters 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        
        # other utils
        self.replay_buffer = BasicBuffer(buffer_maxlen)        
        self.noise = OUNoise(self.env.action_space)
        
    def get_action(self, obs):
        state = torch.FloatTensor([obs]).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        return action
    
    def update(self, batch_size):
        # check data enough
        if len(self.replay_buffer) <= batch_size:
            return 'data not enough!!'
        
        # sample a minibatch data for train
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        # update critic 
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        # update actor
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def train(self,max_episodes,batch_size,render_area,score_area,progress_bar):
        episode_rewards = []
        for episode in range(max_episodes):    
            # init all
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            # not done
            while not done:
                render_area.image(self.env.render(mode='rgb_array'))
                action = self.get_action(state)
                print(action)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward 
                self.update(batch_size)
                state = next_state   
            
            # is done
            episode_rewards.append(episode_reward)
            print("Episode:{} reward:{}".format(episode,episode_reward))      
            row = pd.DataFrame([[episode_reward,np.mean(episode_rewards[-10:])]],
            	columns=['reward','rolling_reward']).astype("float")
            score_area.add_rows(row)
            progress_bar.progress((episode + 1)/max_episodes)            
        
        return episode_rewards

#=====================================================================================================================================================
class TD3Agent:
    def __init__(self, env, gamma=0.99, tau=1e-2, buffer_maxlen=100000, delay_step=2, noise_std=0.2, noise_bound=0.5, critic_lr=1e-3, actor_lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # hyperparameters    
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.update_step = 0 
        self.delay_step = delay_step
        
        # initialize actor and critic networks
        self.critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic1_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        # initialize optimizers        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr) 
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.replay_buffer = BasicBuffer(buffer_maxlen)        

    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        action_space_noise = self.generate_action_space_noise(action_batch)
        next_actions = self.actor.forward(state_batch) + action_space_noise
        next_Q1 = self.critic1_target.forward(next_state_batch, next_actions)
        next_Q2 = self.critic2_target.forward(next_state_batch, next_actions)
        expected_Q = reward_batch + self.gamma * torch.min(next_Q1, next_Q2)

        # critic loss
        curr_Q1 = self.critic1.forward(state_batch, action_batch)
        curr_Q2 = self.critic2.forward(state_batch, action_batch)
        critic1_loss = F.mse_loss(curr_Q1, expected_Q.detach())
        critic2_loss = F.mse_loss(curr_Q2, expected_Q.detach())
        
        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delyaed update for actor & target networks  
        if(self.update_step % self.delay_step == 0):
            # actor
            self.actor_optimizer.zero_grad()
            policy_gradient = -self.critic1(state_batch, self.actor(state_batch)).mean()
            policy_gradient.backward()
            self.actor_optimizer.step()

            # target networks
            self.update_targets()

        self.update_step += 1

    def generate_action_space_noise(self, action_batch):
        noise = torch.normal(torch.zeros(action_batch.size()), self.noise_std).clamp(-self.noise_bound, self.noise_bound).to(self.device)
        return noise

    def update_targets(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

class PPOAgent:
    def __init__(self,env,gamma=0.99,lr=1e-3,clip=0.2,K_epoch=8,buffer_maxsize=100000):
        self.device = "cpu"
        
        # env
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Hyperparamters
        self.gamma = gamma
        self.lr = lr
        self.clip = clip
        self.K_epoch = int(K_epoch)
        
        # critic,init
        self.critic = ppo_critic(self.obs_dim)
        self.critic.apply(self._weights_init)
        
        # actor_new,init
        self.actor_new = ppo_actor(self.obs_dim,self.action_dim)
        self.actor_new.apply(self._weights_init)
        
        # actor_old,sync
        self.actor_old = ppo_actor(self.obs_dim,self.action_dim)
        self.sync()
        
        # opt
        self.actor_optimizer = optim.Adam(self.actor_new.parameters(),lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)
        
        # replay buffer
        self.replay_buffer = BasicBuffer(max_size=int(buffer_maxsize))
        
        # recorder
        self.recorder = {'a_loss':[],
                         'v_loss':[],
                         'ratio':[]}
            
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)
    
    def sync(self):
        for old_param, new_param in zip(self.actor_old.parameters(),self.actor_new.parameters()):
            old_param.data.copy_(new_param.data)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.actor_new.forward(state) # softmax_probs
        dist = Categorical(probs) # Categorical distribution
        act = dist.sample() # smaple action from this Categorical distribution
        return act.detach().item()

    def get_value(state):
        state = torch.FloatTensor(state).to(self.device)
        value = self.critic(state)
        return value.item()

    def update(self,batch_size=64):
        for k in range(self.K_epoch):
            
            # sample mini_batch_data from replay_buffer
            s,a,r,s_,d = self.replay_buffer.sample(batch_size)
            s = torch.FloatTensor(s).to(self.device)
            a = torch.LongTensor(a).to(self.device).unsqueeze(dim=1)
            r = torch.FloatTensor(r).to(self.device)
            s_ = torch.FloatTensor(s_).to(self.device)
            d = torch.FloatTensor(d).to(self.device).unsqueeze(dim=1)
            
            # calculate critic loss
            v_target = r + self.gamma*self.critic(s_)*(1-d)
            v_current = self.critic(s)
            adv =  v_target - v_current
            v_loss = 0.5*(adv**2).mean()
            self.recorder['v_loss'].append(v_loss.item())
            
            # calculate actor loss
            new_p = torch.gather(self.actor_new(s),1,a)
            old_p = torch.gather(self.actor_old(s),1,a)
            ratio = new_p / (old_p + 1e-8)
            self.recorder['ratio'].append(ratio.detach().numpy())
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            surr1 = ratio*adv.detach()
            surr2 = torch.clamp(ratio,1-self.clip,1+self.clip)*adv.detach()
            a_loss = -torch.min(surr1,surr2).mean()
            self.recorder['a_loss'].append(a_loss.item())
            
            # update critic
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()
            
            # update actor
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()
        
        # sync
        self.sync()
            
    def train(self, max_episodes=100, max_steps=10000, batch_size=256, render_area=None, score_area=None, progress_bar=None):
        episode_rewards = []
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                render_area.image(self.env.render(mode='rgb_array'))
                action = self.get_action([state])
                next_state, reward, done, _ = self.env.step(action) 
                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward
                
                if len(self.replay_buffer) >= batch_size:
                    self.update(batch_size)
                    
                if done or step == max_steps-1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    
                    row = pd.DataFrame([[episode_reward,np.mean(episode_rewards[-10:])]],columns=['reward','rolling_reward']).astype("float")
                    score_area.add_rows(row)
                    progress_bar.progress((episode + 1)/max_episodes)
                    
                    if episode_rewards[-1] == 500:
                        print('sloved!!! you get 500 score!')
                        return episode_rewards
                    
                    break
                
                state = next_state
        
        return episode_rewards