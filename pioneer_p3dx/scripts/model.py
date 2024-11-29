from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
from utils import ENV_SETUP, RNN_STATE_PATH, TRAIN_PROBE

class AgentLSTMPolicy(RecurrentMultiInputActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr, **kwargs):
        super().__init__(observation_space, action_space, lr, n_lstm_layers=5, lstm_hidden_size=512, **kwargs)
        if ENV_SETUP == 2:
            self.sensor_set = ['image', 'goal_distance', 'goal_angle']
            # self.sensor_set = ['imu', 'goal_distance', 'goal_angle']
        else:
            self.sensor_set = ['image', 'imu', 'goal_distance', 'goal_angle']

    def forward(self, obs, lstm_states, episode_starts):
        masked_obs = {}
        for key, value in obs.items():
            if key in self.sensor_set:
                masked_obs[key] = value
            else:
                masked_obs[key] = torch.randn_like(value)

        actions, values, log_probs, lstm_states = super().forward(masked_obs, lstm_states, episode_starts)

        if TRAIN_PROBE:
            lstm_states_pi_0 = lstm_states.pi[0].detach().cpu().numpy()
            lstm_states_pi_1 = lstm_states.pi[1].detach().cpu().numpy()
            lstm_states_vf_0 = lstm_states.vf[0].detach().cpu().numpy()
            lstm_states_vf_1 = lstm_states.vf[1].detach().cpu().numpy()

            with open(f'{RNN_STATE_PATH}/lstm_states.npz', 'wb') as f:
                np.savez(f, lstm_states_pi_0=lstm_states_pi_0, lstm_states_pi_1=lstm_states_pi_1, lstm_states_vf_0=lstm_states_vf_0, lstm_states_vf_1=lstm_states_vf_1)

        return actions, values, log_probs, lstm_states


class ProbeLSTMPolicy(RecurrentMultiInputActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr, **kwargs):
        super().__init_(observation_space, action_space, lr, n_lstm_layers=3, **kwargs)
        if ENV_SETUP == 5:
            self.sensor_set = ['image', 'imu', 'goal_distance', 'goal_angle']
        else:
            self.sensor_set = ['imu', 'goal_distance', 'goal_angle']

    def forward(self, obs, lstm_states, episode_starts):
        if episode_starts:
            lstm_states_values = np.load(f'{RNN_STATE_PATH}/lstm_states.npz')
            lstm_states.pi[0] = torch.tensor(lstm_states_values['lstm_states_pi_0']).to(self.device)
            lstm_states.pi[1] = torch.tensor(lstm_states_values['lstm_states_pi_1']).to(self.device)
            lstm_states.vf[0] = torch.tensor(lstm_states_values['lstm_states_vf_0']).to(self.device)
            lstm_states.vf[1] = torch.tensor(lstm_states_values['lstm_states_vf_1']).to(self.device)
            episode_starts = False

        masked_obs = {}
        for key, value in obs.items():
            if key in self.sensor_set:
                masked_obs[key] = value
            else:
                masked_obs[key] = torch.randn_like(value)

        return super().forward(masked_obs, lstm_states, episode_starts)
