"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
Using:
tensorflow 1.0
gym 0.8.0
"""

from simple_emulator import PccEmulator, CongestionControl
from simple_emulator import Packet_selection
from simple_emulator import cal_qoe
import tensorflow as tf
import numpy as np
import random

import gym
import time

np.random.seed(2)
tf.compat.v1.set_random_seed(2)  # reproducible

EVENT_TYPE_FINISHED = 'F'
EVENT_TYPE_DROP = 'D'
EVENT_TYPE_TEMP = 'T'


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.005    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
EPISODE =20
MAX_BANDWITH = 15000

RENDER = False


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + 2), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
    #     return np.random.choice(np.arange(probs.shape[1]), p=temp_p)  # return a int
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

s_dim = 3*EPISODE #speed,losepacket,rtt
a_dim = 3 # +10,0，-5

ddpg = DDPG(a_dim, s_dim)

class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND = False
        self.send_rate = 1000
        self.cwnd = 500
        self.var=3
        self.rtt=0

        self.counter = 0  # EPISODE counter

        self.result_list = []
        self.last_state = []

        for i in range(EPISODE):
            self.last_state.append(100 / MAX_BANDWITH)
        for i in range(EPISODE):
            self.last_state.append(0)
        for i in range(EPISODE):
            self.last_state.append(self.rtt)

    def cc_trigger(self, data):

        event_type = data["event_type"]
        event_time = data["event_time"]
        self.rtt = data["packet_information_dict"]["Latency"]

        if event_type == EVENT_TYPE_DROP:
            self.result_list.append(1)
        else:
            self.result_list.append(0)

        self.counter += 1
        if self.counter == EPISODE:  # choose action every EPISODE times
            self.counter = 0
            print()
            print("EPISODE:")
            print()

            # reward
            r = 0
            for i in range(EPISODE):
                if self.result_list[i] == 0:
                    r += self.send_rate
                else:
                    r += -self.send_rate

            # current_state
            s_ = []
            for i in range(EPISODE):
                s_.append(self.send_rate / MAX_BANDWITH)
            for i in range(EPISODE):
                s_.append(self.result_list[i])
            for i in range(EPISODE):
                s_.append(self.rtt)
            s_array = np.array(s_)

            # choose action and explore
            t = ddpg.choose_action(s_array)
            print(t)
            temp_p= t
            if np.isnan(temp_p[0]):
                temp_p[0] = random.uniform(0, 1.0)
                temp_p[1] = random.uniform(0, 1 - temp_p[0])
                temp_p[2] = 1 - temp_p[0] - temp_p[1]
            a= np.random.choice(np.arange(3), p=temp_p)# return a int
            # a = np.clip(np.random.normal(a, 3), -2, 2)
            # if np.random.uniform() > EPSILON:
            #             #     a = random.randint(0, 2)
            print("action:", a)

            if a == 0:
                self.send_rate += 20.0
            elif a == 1:
                self.send_rate += 0.0
            else:
                self.send_rate += -20.0
                if self.send_rate < 50.0:
                    self.send_rate = 50.0

            # last state
            s = np.array(self.last_state)
            ddpg.store_transition(s, a, r, s_)


            if self.last_state[0] == self.send_rate:
                a = 1
            elif self.last_state[0] > self.send_rate:
                a = 2
            else:
                a = 0

            if ddpg.pointer > MEMORY_CAPACITY:
                self.var *= .9995  # decay the action randomness
                ddpg.learn()

            self.last_state = s_
            self.result_list = []

    def append_input(self, data):
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd": self.cwnd,
                "send_rate": self.send_rate
            }
        return None

class MySolution(Packet_selection, RL):

    def select_packet(self, cur_time, packet_queue):
        """
        The algorithm to select which packet in 'packet_queue' should be sent at time 'cur_time'.
        The following example is selecting packet by the create time firstly, and radio of rest life time to deadline secondly.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#packet_selectionpy.
        :param cur_time: float
        :param packet_queue: the list of Packet.You can get more detail about Block in objects/packet.py
        :return: int
        """

        def is_better(packet):

            best_block_create_time = best_packet.block_info["Create_time"]
            packet_block_create_time = packet.block_info["Create_time"]
            best_block_remainedsize = int(best_packet.block_info["Size"]) // 1480 - int(best_packet.offset)
            packet_remainedsize = int(packet.block_info["Size"]) // 1480 - int(packet.offset)
            best_block_priority = best_packet.block_info["Priority"]
            packet_block_priority = packet.block_info["Priority"]
            # if packet is miss ddl
            if (cur_time - packet_block_create_time) >= packet.block_info["Deadline"]:
                return False
            if (cur_time - best_block_create_time) >= best_packet.block_info["Deadline"]:
                return True
            if best_block_remainedsize == 0:
                return False
            if packet_remainedsize == 0:
                return True
            if (cur_time - best_block_create_time) > best_packet.block_info["Deadline"] / 3 and (
                    cur_time - packet_block_create_time) > \
                    packet.block_info["Deadline"] / 3:
                return (cur_time - best_block_create_time) / best_packet.block_info[
                    "Deadline"] * best_block_priority / (best_block_remainedsize) \
                       > (cur_time - packet_block_create_time) / packet.block_info[
                           "Deadline"] * packet_block_priority / (packet_remainedsize)
            elif (cur_time - best_block_create_time) < best_packet.block_info["Deadline"] / 3:
                return False
            elif (cur_time - packet_block_create_time) < packet.block_info["Deadline"] / 3:
                return True
            else:
                if best_block_priority == packet_block_priority:
                    return best_block_create_time + best_packet.block_info["Deadline"] > packet.block_info[
                        "Deadline"] + packet_block_create_time
                else:
                    return best_block_priority > packet_block_priority

        best_packet_idx = -1
        best_packet = None
        for idx, item in enumerate(packet_queue):
            if best_packet is None or is_better(item):
                best_packet_idx = idx
                best_packet = item

        return best_packet_idx

    def make_decision(self, cur_time):
        """
        The part of algorithm to make congestion control, which will be call when sender need to send pacekt.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#congestion_control_algorithmpy.
        """
        return super().append_input(data)

if __name__ == '__main__':
    # The file path of packets' log
    log_packet_file = "output/packet_log/packet-0.log"

    # Use the object you created above
    my_solution = MySolution()

    # Create the emulator using your solution
    # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/Azson/DTP-emulator/tree/pcc-emulator#constant
    emulator = PccEmulator(
        block_file=["traces/data_video.csv", "traces/data_audio.csv"],
        trace_file="traces/trace.txt",
        solution=my_solution,
        ENABLE_LOG=False
    )

    # Run the emulator and you can specify the time for the emualtor's running.
    # It will run until there is no packet can sent by default.
    emulator.run_for_dur(15)

    # print the debug information of links and senders
    emulator.print_debug()

    print(cal_qoe())