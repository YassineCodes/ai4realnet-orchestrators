class MyAgentWithGNN:
    def __init__(self, gnn_model, actions_path):
        import numpy as np
        self.gnn_model = gnn_model
        self.actions = np.load(actions_path)

    def act(self, observation, reward=0.0, done=False, simulated_act=False):
        x, edge_index = obs_to_graph(observation)
        action_idx = self.gnn_model.act((x, edge_index))
        return self.actions[action_idx]

    def reset(self, observation):
        pass
