import numpy as np
import gym
from gym import spaces
from generator import *
import random
import tensorflow as tf


MAX_REWARD=200 #for instance
MAX_SPEED=1 #NORMALIZED, LETS START WITH THAT

class TrainSystem:

    def __init__(self, T, L, P, gen: Generator):
        self.T = T
        self.L = L
        self.P = P
        self.gen = gen
        self.time = 21600  # 6:00AM
        self.location = np.zeros(gen.trains)
        self.states = []
        for _ in range(self.gen.trains):
            self.states += [TrainState()]
        self.load = np.zeros(gen.trains)
        self.load_before_alight = np.zeros(gen.trains)
        self.platform = np.zeros(gen.stations)
        self.agent_speed = np.zeros(gen.trains)
        self.start_time = [T[train, 0] - L[train, 0] * self.gen.beta[0] for train in range(self.gen.trains)]


    def reward(self):
        return 0

    def new_state_reward(self):
        state = np.concatenate(self.location, self.load, self.platform, axis=0)
        state = np.append(state, self.time)
        reward = self.reward()
        done = (self.states[-1].state == states.FINISHED)
        return state, reward, done

    def reset(self):
        self.time = 21600  # 6:00AM
        self.location = np.zeros(self.gen.trains)
        self.states = []
        for _ in range(self.gen.trains):
            self.states += [TrainState()]
        self.load = np.zeros(self.gen.trains)
        self.load_before_alight = np.zeros(self.gen.trains)
        self.platform = np.zeros(self.gen.stations)
        self.agent_speed = np.zeros(self.gen.trains)

    def Wait(self, train, epoch):
        max_wait = self.start_time[train] - self.time
        if epoch > max_wait:
            self.Load(train, epoch - max_wait)

    def Load(self, train, effective_epoch):
        self.states[train].state = states.LOADING
        if effective_epoch > 0:
            station = self.states[train].station
            potential_load = min(effective_epoch / self.gen.beta[station], self.gen.lmax - self.load[train])
            self.load[train] += min(potential_load, self.platform[station])
            if potential_load < self.platform[station]:
                self.platform[station] -= potential_load
                if self.load[train] == self.gen.lmax:
                    loading_time = (potential_load * self.gen.beta[station])
                    self.Move(train, effective_epoch - loading_time)
            else:
                loading_time = (self.platform[station] * self.gen.beta[station])
                self.platform[station] = 0
                self.Move(train, effective_epoch - loading_time)

    def Unload(self, train, effective_epoch):
        self.states[train].state = states.UNLOADING  # maybe it should be outside, think about it later
        if effective_epoch > 0:
            station = self.states[train].station
            potential_unload = effective_epoch / self.gen.alpha[station]
            max_unload = self.load[train] - self.load_before_alight[train] * (1 - self.gen.eta[train, station])
            self.load[train] -= min(potential_unload, max_unload)
            if potential_unload >= max_unload:
                self.Load(train, effective_epoch - max_unload * self.gen.alpha[station])

    def Move(self, train, effective_epoch):
        self.states[train].state = states.MOVING
        speed = (self.gen.speed_kmh / units.hour) + self.agent_speed[train]
        if(self.states[train].station==self.gen.stations-1):
            self.states[train].state = states.FINISHED
        else:
            if effective_epoch > 0:
                potential_move = effective_epoch * speed
                max_move = (10 - (self.location[train]) % 10)
                moving_distance = min(potential_move, max_move)
                moving_time = moving_distance / speed
                self.location[train] += moving_distance
                if potential_move >= max_move:
                    self.states[train].station += 1
                    self.load_before_alight[train] = self.load[train]
                    self.Unload(train, effective_epoch - moving_time)

    def step(self, epoch=60, noise=0):
        self.time = self.time + epoch
        for i in range(self.gen.stations):
            if self.gen.open_time[i] <= self.time <= self.gen.close_time[i]:
                self.platform[i] = self.platform[i] + (self.gen.lambda_[i] + noise * random.uniform(-0.3, 1.2)) * epoch
        for train in range(self.gen.trains):
            # CASE 0 - Finished
            if self.states[train].state == states.MOVING and self.states[train].station == self.gen.stations - 1:
                self.states[train].state = states.FINISHED
                print("Train")
            elif self.states[train].state == states.WAITING_FOR_FIRST_DEPART:
                self.Wait(train, epoch)
            # CASE 2 - loading
            elif self.states[train].state == states.LOADING:
                self.Load(train, epoch)
            # CASE 3 - Unloading
            elif self.states[train].state == states.UNLOADING:
                self.Unload(train, epoch)
            # CASE 4 - Moving
            elif self.states[train].state == states.MOVING:
                self.Move(train, epoch)
        return self.new_state_reward()

    def state(self):
        return np.concatenate(self.load, self.location, self.platform, np.array(self.time), axis=0)
        #return np.concatenate(np.full(MIN_LOAD, trains) ,np.const_array(MIN_LOCATION) )


class GymTrainSystem(gym.Env):
    def __init__(self, T, L, P, g):
        self.sys = TrainSystem(T, L, P, g)        
        super(GymTrainSystem,self).__init__()
        #reward range:
        MIN_LOCATION=0
        MIN_LOAD=0
        START_TIME=self.sys.gen.open_time
        END_TIME=self.sys.gen.close_time
        MAX_LOAD=self.sys.gen.lmax
        MAX_LOCATION=(self.sys.gen.stations -1 )*10 #I think its -1 but maximum we'll change that later after checking
        #self.reward_range(0,MAX_REWARD)
        
        #action space: 
        self.action_space=spaces.Box(low=0,high=MAX_SPEED,shape=(1,self.sys.gen.trains),dtype=np.float32)
        self.observation_space=spaces.Box(low=np.array([MIN_LOAD,MIN_LOCATION,MIN_LOCATION,0]),high=np.array([MAX_LOAD,MAX_LOCATION,MAX_LOCATION,99999999]))
        #TODO: does problems with START_TIME AND END_TIME 
        
    def reset(self):
        self.sys.reset()

    def step(self, action):
        self.sys.agent_speed = action
        return self.sys.step()

    def render(self, mode='human'):
        pass
    
    

