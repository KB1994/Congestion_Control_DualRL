import random
import requests
import time
import tensorflow as tf
import tf_slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import math

# Helper to fetch stats from Ryu REST API
default_controller_ip = '127.0.0.1'
default_controller_port = 8080
default_dpid = 1

class RyuFetcher:
    def __init__(self, controller_ip=default_controller_ip, port=default_controller_port, dpid=default_dpid):
        self.base = f'http://{controller_ip}:{port}'
        self.dpid = str(dpid)

    def get_port_stats(self, port_no):
        url = f'{self.base}/stats/port/{self.dpid}'
        resp = requests.get(url)
        resp.raise_for_status()
        stats = resp.json().get(self.dpid, [])
        for p in stats:
            if p['port_no'] == port_no:
                return p
n        raise ValueError(f'Port {port_no} not found')

    def get_queue_stats(self, port_no, queue_id=0):
        url = f'{self.base}/stats/queue/{self.dpid}/{port_no}'
        resp = requests.get(url)
        resp.raise_for_status()
        queues = resp.json().get(self.dpid, {}).get(str(port_no), [])
        for q in queues:
            if q['queue_id'] == queue_id:
                return q
        raise ValueError(f'Queue {queue_id} on port {port_no} not found')

# Environment and agent definitions
class Environment:
    def __init__(self):
        self.num_states = 4
        self.num_actions = 2
        self.BANDWIDTH_MAX = 100  # Mbps
        self.BANDWIDTH_MIN = 50   # Mbps
        self.queue_length = 10    # packets
        self.service_rate = 100   # packets/sec
        self.processing_time_per_packet = 0.001  # sec
        self.actions = ['increase_bandwidth', 'decrease_bandwidth']

    def reset(self):
        return 0

    def current_state(self, PACKET_SIZE_c, PACKET_SIZE_d,
                      TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d,
                      Bw_c, Bw_d, packets_received_d, packets_received_c):
        loss_c = (TOTAL_PACKETS_SENT_c - packets_received_c) / TOTAL_PACKETS_SENT_c
        loss_d = (TOTAL_PACKETS_SENT_d - packets_received_d) / TOTAL_PACKETS_SENT_d
        delay_c = PACKET_SIZE_c/(Bw_c * 1e6) + (self.queue_length/self.service_rate) + self.processing_time_per_packet
        delay_d = PACKET_SIZE_d/(Bw_d * 1e6) + (self.queue_length/self.service_rate) + self.processing_time_per_packet
        thrghpt_c = TOTAL_PACKETS_SENT_c / delay_c
        thrghpt_d = TOTAL_PACKETS_SENT_d / delay_d
        return [delay_c, loss_c, delay_d, loss_d], thrghpt_c, thrghpt_d

    def take_action(self, action_n, current_bandwidth):
        selected_action = self.actions[action_n % len(self.actions)]
        if selected_action == 'increase_bandwidth':
            Bw = self.increase_bandwidth(current_bandwidth)
        else:
            Bw = self.decrease_bandwidth(current_bandwidth)
        return Bw, selected_action

    def increase_bandwidth(self, current_bandwidth):
        return min(self.BANDWIDTH_MAX, current_bandwidth + 10)

    def decrease_bandwidth(self, current_bandwidth):
        return max(self.BANDWIDTH_MIN, current_bandwidth - 10)

class DuelingDQNOutput(keras.layers.Layer):
    def call(self, inputs):
        value = inputs[:, :1]
        advantages = inputs[:, 1:]
        return value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))


def create_agent(loss_function, optimizer):
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(1 + 2)  # value + 2 actions
    ])
    model.add(DuelingDQNOutput())
    model.compile(optimizer=optimizer, loss=loss_function)
    return model

# Plot utilities

def learning_plot(total_episodes, in_hist_d, in_hist_c, de_hist_d, de_hist_c):
    plt.figure(figsize=(10,4))
    plt.grid(True, linestyle='--')
    plt.title('Learning Performance')
    plt.plot(range(total_episodes), in_hist_d, label='Inc_D')
    plt.plot(range(total_episodes), in_hist_c, label='Inc_C')
    plt.plot(range(total_episodes), de_hist_d, label='Dec_D')
    plt.plot(range(total_episodes), de_hist_c, label='Dec_C')
    plt.xlabel('Episode'); plt.ylabel('Count'); plt.legend(); plt.savefig('learning.pdf')


def accuracy_plot(total_episodes, th_d, th_c):
    max_t = max(max(th_d), max(th_c), 1)
    plt.figure(figsize=(10,4))
    plt.grid(True, linestyle='--')
    plt.title('Normalized Throughput')
    plt.plot([t/max_t for t in th_d], label='Agent D')
    plt.plot([t/max_t for t in th_c], label='Agent C')
    plt.plot([1]*total_episodes, label='Target')
    plt.xlabel('Episode'); plt.ylabel('Normalized'); plt.legend(); plt.savefig('accuracy.pdf')


def reward_plot(total_episodes, rw_hist):
    plt.figure(figsize=(10,4))
    plt.grid(True, linestyle='--')
    plt.title('Reward')
    plt.plot(rw_hist, label='Reward')
    plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend(); plt.savefig('reward.pdf')

# Main training loop
if __name__ == '__main__':
    # Hyperparameters
    s_size, a_size = 4, 2
    total_episodes = 600
    gamma = 0.95
    max_epsilon, min_epsilon, decay_rate = 1.0, 0.01, 0.005

    env = Environment()
    agent_c = create_agent('mse', keras.optimizers.Adam(0.001))
    agent_d = create_agent('mse', keras.optimizers.Adam(0.001))
    scaler = StandardScaler()
    fetcher = RyuFetcher(controller_ip='127.0.0.1', port=8080, dpid=1)

    # Histories
    in_d, in_c, de_d, de_c = [], [], [], []
    th_d, th_c, rw_hist = [], [], []
    reward_accum = 0

    for e in range(total_episodes):
        # start bandwidth at minimum
        Bw_c = Bw_d = env.BANDWIDTH_MIN
        done = False

        # initial dummy state
        state = np.zeros((1, s_size))
        scaled_state = scaler.fit_transform(state)
        
        num_inc_c = num_dec_c = num_inc_d = num_dec_d = 0

        while not done:
            # fetch real stats
            try:
                stats_c = fetcher.get_port_stats(port_no=1)
                stats_d = fetcher.get_port_stats(port_no=2)
            except Exception as err:
                print("Fetch error:", err)
                time.sleep(1)
                continue

            # compute variables
            TOTAL_PACKETS_SENT_c = stats_c['tx_packets']
            packets_received_c   = stats_c['rx_packets']
            PACKET_SIZE_c = stats_c['tx_bytes'] / max(1, TOTAL_PACKETS_SENT_c)

            TOTAL_PACKETS_SENT_d = stats_d['tx_packets']
            packets_received_d   = stats_d['rx_packets']
            PACKET_SIZE_d = stats_d['tx_bytes'] / max(1, TOTAL_PACKETS_SENT_d)

            # state computation
            state_vals, thr_c, thr_d = env.current_state(
                PACKET_SIZE_c, PACKET_SIZE_d,
                TOTAL_PACKETS_SENT_c, TOTAL_PACKETS_SENT_d,
                Bw_c, Bw_d,
                packets_received_d, packets_received_c
            )
            state = np.array(state_vals).reshape(1, s_size)
            scaled_state = scaler.fit_transform(state)

            # epsilon-greedy
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*e)
            if np.random.rand() < epsilon:
                act_c = np.random.randint(a_size)
                act_d = np.random.randint(a_size)
            else:
                act_c = np.argmax(agent_c.predict(scaled_state)[0])
                act_d = np.argmax(agent_d.predict(scaled_state)[0])

            # apply actions
            Bw_c, sel_c = env.take_action(act_c, Bw_c)
            Bw_d, sel_d = env.take_action(act_d, Bw_d)
            if sel_c == 'increase_bandwidth': num_inc_c += 1
            else: num_dec_c += 1
            if sel_d == 'increase_bandwidth': num_inc_d += 1
            else: num_dec_d += 1

            # termination
            done = (state_vals[1] < 0.03 and state_vals[3] < 0.03 and
                    state_vals[0] < 0.3  and state_vals[2] < 0.3)

            # reward
            reward_accum += 0.5*(num_dec_c + num_dec_d)
            exp_reward = 1 - math.exp(-decay_rate*reward_accum)

            # target computation
            next_scaled = scaler.transform(state)
            target_c = exp_reward + gamma * np.max(agent_c.predict(next_scaled)[0])
            target_d = exp_reward + gamma * np.max(agent_d.predict(next_scaled)[0])

            tf_c = agent_c.predict(scaled_state)
            tf_d = agent_d.predict(scaled_state)
            tf_c[0][act_c] = target_c
            tf_d[0][act_d] = target_d

            agent_c.fit(scaled_state, tf_c, epochs=1, batch_size=64, verbose=0)
            agent_d.fit(scaled_state, tf_d, epochs=1, batch_size=64, verbose=0)

            # record
            th_c.append(thr_c)
            th_d.append(thr_d)
            in_c.append(num_inc_c); de_c.append(num_dec_c)
            in_d.append(num_inc_d); de_d.append(num_dec_d)
            rw_hist.append(exp_reward)

        # end epoch plots
        learning_plot(total_episodes, in_d, in_c, de_d, de_c)
        accuracy_plot(total_episodes, th_d, th_c)
        reward_plot(total_episodes, rw_hist)

        print(f"Episode {e+1}/{total_episodes}, Reward: {exp_reward:.4f}")
