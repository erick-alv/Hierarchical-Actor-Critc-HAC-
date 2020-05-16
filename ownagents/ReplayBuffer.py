import numpy as np
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.index = 0
        self.obs = []
        self.actions = []
        self.obs1 = []
        self.dones = []
        self.rews = []
        self.gammas = []

    def size(self):
        return self.index % self.max_size

    def add(self, ob, action, ob1, done, reward, gamma):
        def _popLeftAll():
            for l in [self.obs, self.actions, self.obs1, self.dones, self.rews, self.gammas]:
                l.pop(0)
        if self.index >= self.max_size:
            _popLeftAll()
        self.index += 1
        self.obs.append(ob)
        self.actions.append(action)
        self.obs1.append(ob1)
        self.dones.append(done)
        self.rews.append(reward)
        self.gammas.append(gamma)

    def sample_batch(self, batch_size):
        obs = []
        actions = []
        obs1 = []
        dones = []
        rws = []
        gammas = []
        idx = np.random.randint(0, min(self.index, self.max_size), batch_size)
        for i in idx:
            obs.append(self.obs[i])
            actions.append(self.actions[i])
            obs1.append(self.obs1[i])
            dones.append(self.dones[i])
            rws.append(self.rews[i])
            gammas.append(self.gammas[i])

        return {
            'obs':obs,
            'actions':actions,
            'obs1':obs1,
            'dones':dones,
            'rewards':rws,
            'gammas':gammas,
        }

    def get_all_as_dict(self):
        obs = []
        actions = []
        obs1 = []
        dones = []
        rws = []
        gammas = []
        for i in range(self.index % self.max_size):
            obs.append(self.obs[i])
            actions.append(self.actions[i])
            obs1.append(self.obs1[i])
            dones.append(self.dones[i])
            rws.append(self.rews[i])
            gammas.append(self.gammas[i])

        return {
            'obs': obs,
            'actions': actions,
            'obs1': obs1,
            'dones': dones,
            'rewards': rws,
            'gammas': gammas,
        }

    def store_from_dict(self, dict):
        n = len(dict['obs'])
        for i in range(n):
            self.add(ob=dict['obs'][i],
                     action=dict['actions'][i],
                     ob1=dict['obs1'][i],
                     done=dict['dones'][i],
                     reward=dict['rewards'][i],
                     gamma=dict['gammas'][i])
