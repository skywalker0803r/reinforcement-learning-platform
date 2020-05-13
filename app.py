import streamlit as st
import pandas as pd
import numpy as np
import gym
from agent.PPO import PPOAgent
from utils import get_agent_params

# config
agent_dict = {'PPO':PPOAgent}
env_list = ['CartPole-v0','Pendulum-v0','LunarLander-v2','LunarLanderContinuous-v2','BipedalWalker-v3']

# main UI
st.title('Reinforcement Learning Platform 2.0')
game_title = st.empty()
render_area = st.empty()
progress_bar = st.progress(0)
score_area = st.line_chart(pd.DataFrame([[np.nan,np.nan]],columns=['reward','rolling_reward']).astype("float"))

# sidebar UI
st.sidebar.subheader('algorithm')
alg_name = st.sidebar.selectbox('',tuple(agent_dict.keys()))

st.sidebar.subheader('environment')
env_name = st.sidebar.selectbox('',tuple(env_list))

st.sidebar.subheader('Hyperparamter')
max_episodes = st.sidebar.number_input('max_episodes',value = 1000)
max_steps = st.sidebar.number_input('max_steps',value = 1000)
batch_size = st.sidebar.number_input('batch_size',value = 64)

# Agent's Hyperparameters
alg_param = {}
for key,value in get_agent_params(agent_dict[alg_name]).items():
    alg_param[key] = st.sidebar.number_input(key,value = value,format = "%.4f" )

# start button
start = st.sidebar.button('start training')
if start:
    env = gym.make(env_name)
    agent = agent_dict[alg_name](env,**alg_param)
    agent.train(max_episodes=max_episodes,max_steps=max_steps,batch_size=batch_size,render_area=render_area,score_area=score_area,progress_bar=progress_bar)