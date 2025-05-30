import random
import requests
import time
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
import math

# Default controller settings
default_controller_ip = '127.0.0.1'
default_controller_port = 8080
default_dpid = 1

class RyuFetcher:
    """
    Helper to fetch stats and apply bandwidth limits via Ryu REST API.
    Assumes the Ryu simple_switch_13_rest_qos app is running.
    """
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
        raise ValueError(f'Port {port_no} not found')

    def get_queue_stats(self, port_no, queue_id=0):
        url = f'{self.base}/stats/queue/{self.dpid}/{port_no}'
        resp = requests.get(url)
        resp.raise_for_status()
        queues = resp.json().get(self.dpid, {}).get(str(port_no), [])
        for q in queues:
            if q['queue_id'] == queue_id:
                return q
        raise ValueError(f'Queue {queue_id} on port {port_no} not found')

    def set_bandwidth_limit(self, port_no, max_rate_mbps, queue_id=0):
        """
        Apply a maximum bandwidth limit (in Mbps) on the given port/queue via REST QoS API.
        """
        url = f'{self.base}/qos/rules/json'
        payload = {
            "dpid": int(self.dpid),
            "port_no": port_no,
            "max_rate": int(max_rate_mbps * 1e6),
            "queue_id": queue_id
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

class Environment:
    def __init__(self, fetcher=None, port_no=None):
        self.fetcher = fetcher
        self.port_no = port_no
        self.BANDWIDTH_MAX = 100  # Mbps
        self.BANDWIDTH_MIN = 50   # Mbps
        self.queue_length = 10    # packets
        self.service_rate = 100   # pkts/sec
        self.processing_time_per_packet = 0.001  # sec
        self.actions = ['increase_bandwidth', 'decrease_bandwidth']

    def reset(self):
        return np.zeros(4)

    def current_state(self, pkt_size_c, pkt_size_d,
                      sent_c, sent_d,
                      bw_c, bw_d,
                      rx_d, rx_c):
        loss_c = (sent_c - rx_c) / sent_c
        loss_d = (sent_d - rx_d) / sent_d
        delay_c = pkt_size_c/(bw_c*1e6) + self.queue_length/self.service_rate + self.processing_time_per_packet
        delay_d = pkt_size_d/(bw_d*1e6) + self.queue_length/self.service_rate + self.processing_time_per_packet
        thr_c = sent_c / delay_c
        thr_d = sent_d / delay_d
        return [delay_c, loss_c, delay_d, loss_d], thr_c, thr_d

    def take_action(self, action_idx, current_bw):
        sel = self.actions[action_idx % len(self.actions)]
        if sel == 'increase_bandwidth':
            new_bw = min(self.BANDWIDTH_MAX, current_bw + 10)
        else:
            new_bw = max(self.BANDWIDTH_MIN, current_bw - 10)
        # Apply via Ryu REST API
        if self.fetcher and self.port_no is not None:
            try:
                self.fetcher.set_bandwidth_limit(self.port_no, new_bw)
            except Exception as e:
                print(f"Failed to apply bandwidth on port {self.port_no}: {e}")
        return new_bw, sel

class DuelingDQNOutput(keras.layers.Layer):
    def call(self, inputs):
        value = inputs[:, :1]
        adv   = inputs[:, 1:]
        return value + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))


def create_agent(loss_fn, optimizer):
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(1 + 2)
    ])
    model.add(DuelingDQNOutput())
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model

# Plot helpers omitted for brevity (same as before) ...

def main():
    # Settings
    s_size, a_size = 4, 2
    episodes = 600
    gamma = 0.95
    eps_max, eps_min, decay = 1.0, 0.01, 0.005

    fetcher_c = RyuFetcher(controller_ip='127.0.0.1', port=8080, dpid=1)
    fetcher_d = RyuFetcher(controller_ip='127.0.0.1', port=8080, dpid=1)
    env_c = Environment(fetcher=fetcher_c, port_no=1)
    env_d = Environment(fetcher=fetcher_d, port_no=2)

    agent_c = create_agent('mse', keras.optimizers.Adam(0.001))
    agent_d = create_agent('mse', keras.optimizers.Adam(0.001))
    scaler = StandardScaler()

    histories = { 'inc_c':[], 'dec_c':[], 'inc_d':[], 'dec_d':[],
                  'thr_c':[], 'thr_d':[], 'rew':[] }
    total_reward = 0

    for e in range(episodes):
        bw_c = bw_d = 50
        done = False
        # initial state
        state = np.zeros((1, s_size))
        for _ in range(1000):  # max steps guard
            # fetch stats
            try:
                st_c = fetcher_c.get_port_stats(1)
                st_d = fetcher_d.get_port_stats(2)
            except:
                time.sleep(0.5)
                continue

            sent_c, rx_c = st_c['tx_packets'], st_c['rx_packets']
            sent_d, rx_d = st_d['tx_packets'], st_d['rx_packets']
            pkt_c = st_c['tx_bytes']/max(1, sent_c)
            pkt_d = st_d['tx_bytes']/max(1, sent_d)

            vals, th_c, th_d = env_c.current_state(pkt_c, pkt_d, sent_c, sent_d, bw_c, bw_d, rx_d, rx_c)
            state = np.array(vals).reshape(1, s_size)
            s_scaled = scaler.fit_transform(state)

            eps = eps_min + (eps_max-eps_min)*math.exp(-decay*e)
            if random.random() < eps:
                a_c = random.randrange(a_size)
                a_d = random.randrange(a_size)
            else:
                a_c = np.argmax(agent_c.predict(s_scaled)[0])
                a_d = np.argmax(agent_d.predict(s_scaled)[0])

            bw_c, act_c = env_c.take_action(a_c, bw_c)
            bw_d, act_d = env_d.take_action(a_d, bw_d)

            done = all([v < thresh for v, thresh in zip(vals, [0.3,0.03,0.3,0.03])])

            # reward & train
            inc = (act_c=='increase_bandwidth') + (act_d=='increase_bandwidth')
            dec = 2-inc
            total_reward += 0.5*dec
            rw = 1-math.exp(-decay*total_reward)

            next_scaled = scaler.transform(state)
            t_c = rw + gamma*np.max(agent_c.predict(next_scaled)[0])
            t_d = rw + gamma*np.max(agent_d.predict(next_scaled)[0])

            tf_c = agent_c.predict(s_scaled)
            tf_d = agent_d.predict(s_scaled)
            tf_c[0][a_c] = t_c
            tf_d[0][a_d] = t_d

            agent_c.fit(s_scaled, tf_c, epochs=1, batch_size=64, verbose=0)
            agent_d.fit(s_scaled, tf_d, epochs=1, batch_size=64, verbose=0)

            # record
            histories['inc_c'].append(int(act_c=='increase_bandwidth'))
            histories['dec_c'].append(int(act_c=='decrease_bandwidth'))
            histories['inc_d'].append(int(act_d=='increase_bandwidth'))
            histories['dec_d'].append(int(act_d=='decrease_bandwidth'))
            histories['thr_c'].append(th_c)
            histories['thr_d'].append(th_d)
            histories['rew'].append(rw)

            if done: break

        # optional: call plot funcs here
        print(f"Episode {e+1}/{episodes}, reward={rw:.4f}")

if __name__=='__main__':
    main()
