import numpy as np
from pr3_utils import load_data, inversePose, projection, projectionJacobian, visualize_trajectory_2d, axangle2pose, pose2adpose
from part_1 import ekf_prediction
from part_3 import stereo_measurement_function, stero_measurement_jacobian, initialize_landmark_from_measurement
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse as sp

def visual_inertial_slam(v_t, w_t, features, timestamps, 
                         extL_T_imu, extR_T_imu, 
                         K_l, K_r, 
                         Q_imu=1e-3,       # Increase IMU process noise to allow more visual update correction
                         Q_landmark=1e2,   # Initial uncertainty for landmark initialization
                         R_meas=2.0,       # Measurement noise
                         reproj_error_thresh=40.0,  # Increase threshold to retain more valid measurements
                         max_landmarks_per_update=100):  # Increase the number of landmarks used per update
        
    # 1. Basic Initialization
    n_timestamps = len(timestamps)
    n_landmarks = features.shape[1]
    
    # IMU pose state (position + orientation in SE(3))
    T_array = np.zeros((n_timestamps, 4, 4))
    T_array[0] = np.eye(4)  # initial pose
    
    # Landmark states
    landmarks = np.zeros((n_landmarks, 3))
    landmarks_initialized = np.zeros(n_landmarks, dtype=bool)
    
    # Covariance matrices
    # Pose covariance: 6x6 (for 3 translations and 3 rotations)
    P_pose = np.eye(6) * 1e-6  # Increase initial uncertainty slightly
    
    # Landmarks covariance: maintain a 3x3 covariance matrix for each landmark
    P_landmarks = np.zeros((n_landmarks, 3, 3))
    for i in range(n_landmarks):
        P_landmarks[i] = np.eye(3) * Q_landmark
    
    # Measurement noise matrix (for each 4D measurement)
    R = np.eye(4) * R_meas
    

    
    # 2. Main Loop: From t=1, perform IMU prediction + stereo visual update
    for t in tqdm(range(1, n_timestamps)):
        dt = timestamps[t] - timestamps[t-1]
        
        # === (A) IMU Prediction Step ===
        twist = np.hstack((v_t[t-1], w_t[t-1]))  # [vx, vy, vz, wx, wy, wz]
        delta_T = axangle2pose(twist * dt)         # Convert se(3) to SE(3) using exponential map
        
        # Update IMU pose
        T_array[t] = T_array[t-1] @ delta_T
        
        # Update pose covariance (using simplified model)
        F = np.eye(6)                 # State transition matrix
        G = np.eye(6) * dt            # Control input matrix
        Q = np.eye(6) * Q_imu         # Process noise
        P_pose = F @ P_pose @ F.T + G @ Q @ G.T
        
        # === (B) Vision Update Step ===
        # Identify indices of landmarks visible in the current frame
        visible_indices = np.where(np.any(features[:, :, t] != -1, axis=0))[0]
        if len(visible_indices) == 0:
            continue  # No features available
        
        # To reduce computational load, limit the number of landmarks per update
        if len(visible_indices) > max_landmarks_per_update:
            # Prioritize already-initialized landmarks
            initialized_visible = visible_indices[landmarks_initialized[visible_indices]]
            uninitialized_visible = visible_indices[~landmarks_initialized[visible_indices]]
            
            n_init = len(initialized_visible)
            n_uninit_to_select = min(max_landmarks_per_update - n_init, len(uninitialized_visible))
            
            if n_uninit_to_select > 0:
                selected_uninit = np.random.choice(uninitialized_visible, n_uninit_to_select, replace=False)
                selected_indices = np.concatenate((initialized_visible, selected_uninit))
            else:
                # If there are more than max_landmarks_per_update already-initialized, choose a random subset
                selected_indices = np.random.choice(initialized_visible, max_landmarks_per_update, replace=False)
        else:
            selected_indices = visible_indices
        
        # Initialize uninitialized landmarks using the current frame
        newly_initialized = 0
        for idx in selected_indices:
            if not landmarks_initialized[idx]:
                z_i = features[:, idx, t]  # [ul, vl, ur, vr]
                if np.any(z_i == -1):  # Ensure measurement is valid
                    continue
                    
                m_init = initialize_landmark_from_measurement(
                    z_i, T_array[t], extL_T_imu, extR_T_imu, K_l, K_r
                )
                # Additional check: ensure initialization is reasonable (not NaN and within expected bounds)
                if not np.isnan(m_init).any() and np.all(np.abs(m_init) < 200):
                    landmarks[idx] = m_init
                    landmarks_initialized[idx] = True
                    newly_initialized += 1
        
        
        
        # Only use initialized landmarks for update
        update_indices = selected_indices[landmarks_initialized[selected_indices]]
        if len(update_indices) == 0:
            continue
        
        # Build measurement vector, predicted measurement vector, and Jacobians
        n_upd = len(update_indices)
        z_measured = np.zeros(4 * n_upd)
        z_predicted = np.zeros(4 * n_upd)
        H_pose = np.zeros((4 * n_upd, 6))       # Jacobian w.r.t. pose: (4*n_upd) x 6
        H_landmarks = np.zeros((4 * n_upd, 3 * n_upd))  # Jacobian w.r.t. each landmark (block diagonal)
        
        valid_measurements = []
        total_reprojection_error = 0
        
        for i, idx in enumerate(update_indices):
            z_i_measured = features[:, idx, t]
            if np.any(z_i_measured == -1):
                continue
                
            z_measured[4*i:4*i+4] = z_i_measured
            
            # Predict measurement using current pose and landmark state
            m_i = landmarks[idx]
            z_i_pred = stereo_measurement_function(m_i, T_array[t], 
                                                   extL_T_imu, extR_T_imu, 
                                                   K_l, K_r)
            z_predicted[4*i:4*i+4] = z_i_pred
            
            # Compute reprojection error
            reprojection_error = np.sum((z_i_measured - z_i_pred)**2)
            total_reprojection_error += reprojection_error
            
            if reprojection_error > reproj_error_thresh:
                continue  # Treat as outlier and skip
            
            valid_measurements.append(i)
            
            # (1) Landmark part of the Jacobian
            H_lmk_i = stero_measurement_jacobian(m_i, T_array[t], 
                                               extL_T_imu, extR_T_imu, 
                                               K_l, K_r)
            H_landmarks[4*i:4*i+4, 3*i:3*i+3] = H_lmk_i
            
            # (2) Pose part of the Jacobian - using numerical differentiation to compute full 6x derivative
            epsilon = 1e-6
            H_pose_i = np.zeros((4, 6))
            for j in range(6):
                delta = np.zeros(6)
                delta[j] = epsilon
                delta_pose = axangle2pose(delta)
                
                # Perturbed pose
                T_perturbed = T_array[t] @ delta_pose
                
                # Predict measurement with perturbed pose
                z_perturbed = stereo_measurement_function(m_i, T_perturbed, 
                                                        extL_T_imu, extR_T_imu, 
                                                        K_l, K_r)
                
                # Numerical gradient (finite differences)
                H_pose_i[:, j] = (z_perturbed - z_i_pred) / epsilon
            
            H_pose[4*i:4*i+4, :] = H_pose_i
        
        # Record the number of valid measurements
        n_valid = len(valid_measurements)
        
        # If no valid measurement, skip update
        if n_valid == 0:
            continue
        
        # Only retain rows corresponding to valid measurements
        valid_rows = []
        for vm_i in valid_measurements:
            valid_rows.extend(range(4*vm_i, 4*vm_i+4))
        valid_rows = np.array(valid_rows)
        
        z_measured = z_measured[valid_rows]
        z_predicted = z_predicted[valid_rows]
        H_pose = H_pose[valid_rows, :]
        H_landmarks = H_landmarks[valid_rows, :]
        
        # Compute residual
        residual = z_measured - z_predicted
        
        # Full measurement noise matrix
        R_full = np.eye(len(residual)) * R_meas
        
        # === (C) Pose Update ===
        # Increase numerical stability
        S = H_pose @ P_pose @ H_pose.T + R_full
        S = (S + S.T) / 2  # Ensure symmetry  
        S_inv = np.linalg.inv(S)
        S += np.eye(S.shape[0]) * 1e-3
        S_inv = np.linalg.inv(S)
        
        K_pose = P_pose @ H_pose.T @ S_inv
        
        pose_update = K_pose @ residual  # Shape (6,)
        
        # Limit the magnitude of the update
        pose_update = np.clip(pose_update, -0.5, 0.5)
        
        # Convert pose_update from se(3) to SE(3)
        delta_T_update = axangle2pose(pose_update)
        T_array[t] = T_array[t] @ delta_T_update
        
        # Update pose covariance using the Joseph form for stability
        I6 = np.eye(6)
        K_H = K_pose @ H_pose
        P_pose = (I6 - K_H) @ P_pose @ (I6 - K_H).T + K_pose @ R_full @ K_pose.T
        P_pose = (P_pose + P_pose.T) / 2  # Ensure symmetry
        
        # === (D) Landmark Update ===
        for i, idx in enumerate(update_indices):
            if i not in valid_measurements:
                continue
            
            vm_i = valid_measurements.index(i)
            start_idx = 4 * vm_i
            end_idx = start_idx + 4
            
            z_i_measured = z_measured[start_idx:end_idx]
            z_i_predicted = z_predicted[start_idx:end_idx]
            H_lmk_i = H_landmarks[start_idx:end_idx, 3*i:3*i+3]
            
            # Residual for this landmark
            residual_i = z_i_measured - z_i_predicted
            
            # Kalman Gain for the landmark update
            S_i = H_lmk_i @ P_landmarks[idx] @ H_lmk_i.T + R
            S_i = (S_i + S_i.T) / 2  # Ensure symmetry
            
            S_i_inv = np.linalg.inv(S_i)

            
            K_lmk_i = P_landmarks[idx] @ H_lmk_i.T @ S_i_inv
            
            lmk_update = K_lmk_i @ residual_i
            lmk_update = np.clip(lmk_update, -5.0, 5.0)  # Avoid excessively large updates
            landmarks[idx] = landmarks[idx] + lmk_update
            
            
            I3 = np.eye(3)
            K_H_i = K_lmk_i @ H_lmk_i
            P_landmarks[idx] = (I3 - K_H_i) @ P_landmarks[idx] @ (I3 - K_H_i).T + K_lmk_i @ R @ K_lmk_i.T
            P_landmarks[idx] = (P_landmarks[idx] + P_landmarks[idx].T) / 2
    
    return T_array, landmarks, P_landmarks, landmarks_initialized

def visualize_slam_results(T_array, landmarks, landmarks_initialized, path_name="SLAM Trajectory", threshold=200):
    
    fig, ax = visualize_trajectory_2d(T_array, path_name=path_name, show_ori=False)
    
    
    landmarks_valid = landmarks[landmarks_initialized]
    landmarks_valid = landmarks_valid[~np.isnan(landmarks_valid).any(axis=1)]
    
    
    inlier_mask = (np.abs(landmarks_valid).max(axis=1) < threshold)
    inliers = landmarks_valid[inlier_mask]
    
    ax.scatter(inliers[:, 0], inliers[:, 1], marker='o', s=5, color='blue', alpha=0.5, label='Landmarks (Inliers)')
    
    ax.set_title(f"Dataset{num} Visual-Inertial SLAM Results")
    ax.legend()
    ax.grid(True)
    return fig, ax


if __name__ == '__main__':
    num = "00"
    file_name = f"/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR3/data/dataset{num}/dataset{num}.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(file_name)
    
    # Run Visual-Inertial SLAM with modified parameters
    print("Running Visual-Inertial SLAM...")
    T_array, landmarks, P_landmarks, landmarks_initialized = visual_inertial_slam(
        v_t, w_t, features, timestamps,
        extL_T_imu, extR_T_imu, K_l, K_r,
        Q_imu=1e-3,         # Increased IMU process noise
        Q_landmark=1e2,     # Reasonable initial landmark uncertainty
        R_meas=2.0,         # Appropriate measurement noise
        reproj_error_thresh=40.0,  # More lenient threshold to retain more valid measurements
        max_landmarks_per_update=100  # More landmarks per update
    )
    
    # Visualize SLAM results
    print("Visualizing results...")
    fig, ax = visualize_slam_results(T_array, landmarks, landmarks_initialized, 
                                     path_name=f"Dataset {num} SLAM Trajectory",threshold=200)
    ax.legend(fontsize=20)
    
    # Compare with IMU-only prediction
    print("Comparing with IMU-only prediction...")
    T_array_imu_only = ekf_prediction(v_t, w_t, timestamps)
    fig2, ax2 = visualize_trajectory_2d(T_array_imu_only, path_name="IMU-only Trajectory", show_ori=False)
    
    # Plot SLAM trajectory on the same plot for comparison
    ax2.plot(T_array[:, 0, 3], T_array[:, 1, 3], 'g-', label="SLAM Trajectory")
    ax2.set_title(f"Dataset {num} - Comparison of Trajectories")
    ax2.legend(fontsize=20)
    ax2.legend()
    
    plt.show()