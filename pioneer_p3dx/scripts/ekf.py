#!/usr/bin/env python3

import numpy as np
from scipy.linalg import block_diag

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.Q = process_noise
        self.R = measurement_noise

    def predict(self, dt):
        # State transition function
        F = np.eye(len(self.state))
        F[0::6, 3::6] = np.eye(3) * dt
        F[1::6, 4::6] = np.eye(3) * dt
        F[2::6, 5::6] = np.eye(3) * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, measurement):
        # Measurement function (identity for point cloud)
        H = np.eye(len(self.state))

        # Calculate Kalman gain
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        y = measurement - H @ self.state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(len(self.state))
        self.covariance = (I - K @ H) @ self.covariance

    def get_prediction(self, future_steps, dt):
        predicted_states = []
        current_state = self.state.copy()
        F = np.eye(len(self.state))
        F[0::6, 3::6] = np.eye(3) * dt
        F[1::6, 4::6] = np.eye(3) * dt
        F[2::6, 5::6] = np.eye(3) * dt

        for _ in range(future_steps):
            current_state = F @ current_state
            predicted_states.append(current_state)

        return predicted_states