from collections import namedtuple, deque, defaultdict
import random
import torch
import pickle
import numpy as np


class ReplayBuffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size=None, batch_size=None, seed=42, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
        #    self.device)
        #dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        states = defaultdict(list)
        next_states = defaultdict(list)
        actions = []
        rewards = []
        dones = []
        for e in experiences:
            for k, v in e.state.items():
                if k == "vision":
                    states[k].append(np.expand_dims(np.array(v), 0))
                else:
                    states[k].append(np.expand_dims(np.array(v), 0))
            actions.append(np.expand_dims(e.action, 0))
            rewards.append(np.expand_dims(e.reward, 0))
            for k, v in e.next_state.items():
                if k == "vision":
                    next_states[k].append(np.expand_dims(np.array(v), 0))
                else:
                    next_states[k].append(np.expand_dims(np.array(v), 0))
            dones.append(np.expand_dims(int(e.done), 0))

        for k, v in states.items():
            states[k] = torch.from_numpy(np.vstack(v)).float()
        for k, v in next_states.items():
            next_states[k] = torch.from_numpy(np.vstack(v)).float()

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def save(self, path):
        pickle.dump([tuple(e) for e in self.memory], open(path, "wb"))

    @classmethod
    def load(cls, path, **kwargs):
        replay_buffer = cls(**kwargs)
        memory = pickle.load(open(path, "rb"))
        replay_buffer.memory = deque([replay_buffer.experience(*m) for m in memory],
                                     maxlen=replay_buffer.buffer_size)
        return replay_buffer

    def __len__(self):
        return len(self.memory)