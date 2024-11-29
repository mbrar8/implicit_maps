#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from ekf import ExtendedKalmanFilter

class PointCloudVisualizer(Node):
    def __init__(self):
        super().__init__('point_cloud_visualizer')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/depth_camera/points',
            self.point_cloud_callback,
            10)
        self.ekf = None
        self.prediction_steps = 5
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.predicted_pcds = [o3d.geometry.PointCloud() for _ in range(self.prediction_steps)]
        self.setup_visualization()

    def setup_visualization(self):
        self.vis.add_geometry(self.pcd)
        for pred_pcd in self.predicted_pcds:
            self.vis.add_geometry(pred_pcd)

    def point_cloud_callback(self, depth_pc):
        # Process the depth point cloud data
        data = list(pc2.read_points(depth_pc, skip_nans=True, field_names=("x", "y", "z")))
        points = np.array(data)

        if self.ekf is None:
            self.initialize_ekf(points)
        else:
            self.update_ekf(points)

        # Update input point cloud
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.paint_uniform_color([1, 0, 0])  # Red color for input

        # Get and update predicted point clouds
        predictions = self.get_future_predictions()
        for i, pred in enumerate(predictions):
            self.predicted_pcds[i].points = o3d.utility.Vector3dVector(pred)
            self.predicted_pcds[i].paint_uniform_color([0, 1, 0])  # Green color for predictions

        self.vis.update_geometry(self.pcd)
        for pred_pcd in self.predicted_pcds:
            self.vis.update_geometry(pred_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def initialize_ekf(self, points):
        initial_state = np.concatenate([points.flatten(), np.zeros_like(points.flatten())])
        n_points = len(points)
        initial_covariance = np.eye(6 * n_points) * 0.1
        process_noise = np.eye(6 * n_points) * 0.01
        measurement_noise = np.eye(3 * n_points) * 0.1
        self.ekf = ExtendedKalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise)

    def update_ekf(self, points):
        measurement = points.flatten()
        self.ekf.update(measurement)
        self.ekf.predict(dt=0.1)  # Assume 10 Hz update rate

    def get_future_predictions(self):
        if self.ekf is None:
            return None
        predictions = self.ekf.get_prediction(self.prediction_steps, dt=0.1)
        return [pred.reshape(-1, 6)[:, :3] for pred in predictions]

def main(args=None):
    rclpy.init(args=args)
    point_cloud_visualizer = PointCloudVisualizer()
    rclpy.spin(point_cloud_visualizer)
    point_cloud_visualizer.vis.destroy_window()
    point_cloud_visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()