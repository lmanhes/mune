import random

from mune.memory import ReplayBuffer
from mune.representation import MultimodalBetaVAE, MultimodalBetaVAEConfig


class AlphaExplorer(object):

    def __init__(self, actions, memory_path):
        self.actions = actions
        self.memory_path = memory_path

        self.memory = ReplayBuffer(buffer_size=3000, batch_size=32)

        representation_config = MultimodalBetaVAEConfig()
        self.representation_module = MultimodalBetaVAE(config=representation_config)

        self.last_observation = None
        self.last_reward = None
        self.last_action = None

    def random_act(self, observations, reward):
        if self.last_observation and self.last_action:
            self.memory.add_experience(states=self.last_observation,
                                       actions=self.last_action,
                                       rewards=reward,
                                       next_states=observations,
                                       dones=False)
        action = random.choice(self.actions)

        self.last_observation = observations
        self.last_reward = reward
        self.last_action = action

        return action

    def save_memory(self):
        self.memory.save(self.memory_path)