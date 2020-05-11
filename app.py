import streamlit as st
import pandas as pd
import numpy as np
import gym
from agent import DQNAgent,A2CAgent,DDPGAgent,TD3Agent,PPOAgent
from utils import get_agent_params

agent_dict = {
				'DQN':DQNAgent,
				'A2C':A2CAgent,
				'DDPG':DDPGAgent,
				'TD3':TD3Agent,
				'PPO':PPOAgent,
				}

# UI
st.title('Reinforcement Learning Platform')
game_title = st.empty()
render_area = st.empty()
progress_bar = st.progress(0)
score_area = st.line_chart(pd.DataFrame([[np.nan,np.nan]],
	columns=['reward','rolling_reward']).astype("float"))

# select algorithm
st.sidebar.subheader('algorithm')
alg_name = st.sidebar.selectbox('',tuple(agent_dict.keys()))

# select environment
st.sidebar.subheader('environment')
env_name = st.sidebar.selectbox('',('CartPole-v0','Pendulum-v0','LunarLander-v2',
	'LunarLanderContinuous-v2','BipedalWalker-v3'))

# commom Hyperparamter
st.sidebar.subheader('Hyperparamter')
max_episodes = st.sidebar.number_input('max_episodes',value=1000)
batch_size = st.sidebar.number_input('batch_size',value=32)

# algorithm Hyperparamter
alg_param = {}
for k,v in get_agent_params(agent_dict[alg_name]).items():
	alg_param[k] = st.sidebar.number_input(k,value=v,format="%.4f")

# start training
start = st.sidebar.button('start training')
if start:
	env = gym.make(env_name)
	agent = agent_dict[alg_name](env,**alg_param)
	agent.train(max_episodes=max_episodes,batch_size=batch_size,render_area=render_area,
		score_area=score_area,progress_bar=progress_bar)




