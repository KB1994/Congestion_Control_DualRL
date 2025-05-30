import random
import requests
import time
import datetime
import math
from collections import deque, Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================= Constants & Helpers =============================
CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Ryu REST API Helper
class RyuFetcher:
    def __init__(self, controller_ip='127.0.0.1', port=8080, dpid=1):
        self.base = f'http://{controller_ip}:{port}'
        self.dpid = str(dpid)
    def get_port_stats(self, port_no):
        url = f'{self.base}/stats/port/{self.dpid}'
        resp = requests.get(url); resp.raise_for_status()
        for p in resp.json().get(self.dpid, []):
            if p['port_no'] == port_no:
                return p
        raise ValueError(f'Port {port_no} not found')
    def set_bandwidth_limit(self, port_no, mbps, queue_id=0):
        url = f'{self.base}/qos/rules/json'
        payload = {"dpid": int(self.dpid), "port_no": port_no,
                   "max_rate": int(mbps*1e6), "queue_id": queue_id}
        resp = requests.post(url, json=payload); resp.raise_for_status()
        return resp.json()

# Environment wrapper
class Environment:
    def __init__(self, fetcher: RyuFetcher, port_no: int):
        self.fetcher = fetcher; self.port_no = port_no
        self.BW_MAX = 100; self.BW_MIN = 50
        self.queue_len = 10; self.service_rate = 100; self.proc_time = 0.001
        self.actions = ['increase', 'decrease']
    def state_and_metrics(self, pkt_sz_c, pkt_sz_d, sent_c, sent_d, bw_c, bw_d, rx_d, rx_c):
        loss_c = (sent_c - rx_c)/sent_c; loss_d = (sent_d-rx_d)/sent_d
        delay_c = pkt_sz_c/(bw_c*1e6)+self.queue_len/self.service_rate+self.proc_time
        delay_d = pkt_sz_d/(bw_d*1e6)+self.queue_len/self.service_rate+self.proc_time
        thr_c = sent_c/delay_c; thr_d = sent_d/delay_d
        return np.array([delay_c, loss_c, delay_d, loss_d]), thr_c, thr_d
    def apply(self, action_idx, current_bw):
        act = self.actions[action_idx%2]
        new_bw = min(self.BW_MAX, current_bw+10) if act=='increase' else max(self.BW_MIN, current_bw-10)
        try:
            self.fetcher.set_bandwidth_limit(self.port_no, new_bw)
        except Exception as e:
            print(f"QoS apply fail port{self.port_no}", e)
        return new_bw, act

# ============================= Agent Builders =============================
def build_dueling_dqn(input_dim, n_actions):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(1+n_actions)(x)
    # split value & advantage
    def dueling(x):
        v = x[:, :1]; a = x[:, 1:]
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    out = layers.Lambda(dueling)(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return model

# Simplified Rainbow uses noisy layers + dueling
class NoisyDense(layers.Layer):
    def __init__(self, units): super().__init__(); self.units=units
    def build(self, inp_shape):
        self.w_mu = self.add_weight((inp_shape[-1],self.units)); self.w_sigma = self.add_weight((inp_shape[-1],self.units))
        self.b_mu = self.add_weight((self.units,)); self.b_sigma = self.add_weight((self.units,))
    def call(self, x):
        w = self.w_mu + self.w_sigma*tf.random.normal(self.w_mu.shape)
        b = self.b_mu + self.b_sigma*tf.random.normal(self.b_mu.shape)
        return tf.matmul(x,w)+b

def build_rainbow(input_dim,n_actions):
    inp=layers.Input((input_dim,)); x=NoisyDense(64)(inp); x=layers.ReLU()(x)
    x=NoisyDense(64)(x); x=layers.ReLU()(x)
    x=NoisyDense(128)(x); x=layers.ReLU()(x)
    x=layers.Dense(1+n_actions)(x)
    out=layers.Lambda(lambda v: v[:, :1] + (v[:,1:]-tf.reduce_mean(v[:,1:],axis=1,keepdims=True)))(x)
    m=models.Model(inp,out); m.compile(optimizer=keras.optimizers.RMSprop(1e-3),'mse'); return m

# A2C actor-critic builder
def build_a2c(input_dim, n_actions):
    inp=layers.Input((input_dim,)); x=layers.Dense(64,activation='relu')(inp)
    x=layers.Dense(64,activation='relu')(x)
    pi=layers.Dense(n_actions,activation='softmax')(x)
    v=layers.Dense(1)(x)
    actor=models.Model(inp,pi); critic=models.Model(inp,v)
    actor.compile(optimizer='rmsprop','categorical_crossentropy')
    critic.compile(optimizer='rmsprop','mse')
    return actor, critic

# DDPG builder (actor & critic)
def build_ddpg(input_dim):
    # actor continuous [-1,1]
    inp=layers.Input((input_dim,)); x=layers.Dense(64,activation='relu')(inp)
    x=layers.Dense(64,activation='relu')(x); x=layers.Dense(128,activation='relu')(x)
    mu=layers.Dense(1,activation='tanh')(x)
    actor=models.Model(inp,mu); actor.compile('mse')
    # critic
    a_in=layers.Input((1,)); c_in=layers.Concatenate()([inp,a_in])
    y=layers.Dense(64,activation='relu')(c_in); y=layers.Dense(64,activation='relu')(y)
    y=layers.Dense(128,activation='relu')(y); q=layers.Dense(1)(y)
    critic=models.Model([inp,a_in],q); critic.compile('mse')
    return actor, critic

# ============================= Training Pipeline =============================
def train_model(name, model_objs, env_c, env_d, episodes=600):
    scaler=StandardScaler(); history={'reward':[]}
    for e in range(episodes):
        bw_c=bw_d=env_c.BW_MIN; total_r=0; done=False
        state=np.zeros((1,4))
        while not done:
            # fetch
            stc=env_c.fetcher.get_port_stats(env_c.port_no)
            std=env_d.fetcher.get_port_stats(env_d.port_no)
            sent_c,rx_c=stc['tx_packets'],stc['rx_packets']; sent_d,rx_d=std['tx_packets'],std['rx_packets']
            pkt_c, pkt_d = stc['tx_bytes']/sent_c, std['tx_bytes']/sent_d
            s, thr_c, thr_d = env_c.state_and_metrics(pkt_c,pkt_d,sent_c,sent_d,bw_c,bw_d,rx_d,rx_c)
            s_scaled=scaler.fit_transform(s.reshape(1,4))
            # choose action
            if name in ['dueling','rainbow']:
                qvals_c=model_objs['c'].predict(s_scaled); a_c=np.argmax(qvals_c[0])
                qvals_d=model_objs['d'].predict(s_scaled); a_d=np.argmax(qvals_d[0])
            elif name=='a2c':
                p_c=model_objs['actor_c'].predict(s_scaled); a_c=np.random.choice(len(p_c[0]),p=p_c[0])
                p_d=model_objs['actor_d'].predict(s_scaled); a_d=np.random.choice(len(p_d[0]),p=p_d[0])
            elif name=='ddpg':
                cont1=model_objs['actor_c'].predict(s_scaled); a_c=0 if cont1<0 else 1
                cont2=model_objs['actor_d'].predict(s_scaled); a_d=0 if cont2<0 else 1
            # apply
            bw_c,act_c=env_c.apply(a_c,bw_c); bw_d,act_d=env_d.apply(a_d,bw_d)
            # reward
            dec=(act_c=='decrease')+(act_d=='decrease'); total_r+=0.5*dec
            # break
            done= all(s < t for s,t in zip(s,[0.3,0.03,0.3,0.03]))
        history['reward'].append(1-math.exp(-0.005*total_r))
        print(f"{name} ep{e+1}/{episodes} r={history['reward'][-1]:.3f}")
    return history

# ============================= Main Execution =============================
if __name__=='__main__':
    # instantiate fetchers and envs
    fetch=RyuFetcher(); env_c=Environment(fetch,1); env_d=Environment(fetch,2)
    # build all models
    duel_c, duel_d = build_dueling_dqn(4,2), build_dueling_dqn(4,2)
    rain_c, rain_d = build_rainbow(4,2), build_rainbow(4,2)
    a2c_c, v2c = build_a2c(4,2); a2c_d, v2d = build_a2c(4,2)
    ddpg_ac, ddpg_cr = build_ddpg(4); ddpg_ad, ddpg_c2 = build_ddpg(4)
    # train
    histories={}
    histories['dueling']=train_model('dueling',{'c':duel_c,'d':duel_d},env_c,env_d)
    histories['rainbow']=train_model('rainbow',{'c':rain_c,'d':rain_d},env_c,env_d)
    histories['a2c']=train_model('a2c',{'actor_c':a2c_c,'actor_d':a2c_d},env_c,env_d)
    histories['ddpg']=train_model('ddpg',{'actor_c':ddpg_ac,'actor_d':ddpg_ad},env_c,env_d)
    # plot comparison
    plt.figure();
    for k,h in histories.items(): plt.plot(h['reward'],label=k)
    plt.legend(); plt.title('Comparative Reward'); plt.savefig(f'comp_rewards_{CURRENT_TIME}.png')
