import numpy as np
from pr3_utils import load_data, visualize_trajectory_2d, inversePose
from part_1 import ekf_prediction
from tqdm import tqdm
import matplotlib.pyplot as plt

def initialize_landmark_from_measurement(
    z,                    # (ul, vl, ur, vr)
    T_imu_to_world,       # 4x4, IMU->World transform
    extL_T_imu,           # 4x4, IMU->LeftCam transform
    extR_T_imu,           # 4x4, IMU->RightCam transform
    K_l, K_r,             # 3x3 intrinsics (left, right)
    eps=1e-6,
    max_distance=1000
):
    """
    Triangulate a single landmark's 3D position in the world frame using
    linear triangulation with stereo measurements. Includes basic checks
    such as reprojection error and distance filtering.

    Returns:
      A 3D point (x,y,z) in world coordinates, or [NaN, NaN, NaN] if invalid.
    """

    # Unpack stereo measurements
    ul, vl, ur, vr = z

    # 1) World->IMU
    T_world_to_imu = inversePose(T_imu_to_world[np.newaxis, ...])[0]

    # 2) World->LeftCam, World->RightCam
    T_world_to_left  = extL_T_imu @ T_world_to_imu
    T_world_to_right = extR_T_imu @ T_world_to_imu

    # 3) Projection matrices: P_l = K_l * (World->LeftCam[:3,:])
    P_l = K_l @ T_world_to_left[:3, :]
    P_r = K_r @ T_world_to_right[:3, :]

    # 4) Build a system A x = b for linear triangulation
    #    A has shape (4,3), b has shape (4,)
    A = np.zeros((4, 3))
    b = np.zeros(4)

    # Left camera rows
    A[0, :] = ul * P_l[2, :3] - P_l[0, :3]
    A[1, :] = vl * P_l[2, :3] - P_l[1, :3]
    b[0]    = P_l[0, 3] - ul * P_l[2, 3]
    b[1]    = P_l[1, 3] - vl * P_l[2, 3]

    # Right camera rows
    A[2, :] = ur * P_r[2, :3] - P_r[0, :3]
    A[3, :] = vr * P_r[2, :3] - P_r[1, :3]
    b[2]    = P_r[0, 3] - ur * P_r[2, 3]
    b[3]    = P_r[1, 3] - vr * P_r[2, 3]

    
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag([1.0/s if s > eps else 0.0 for s in S])
    x_3 = Vh.T @ S_inv @ U.T @ b
    

    # x_3 is the 3D position in world coords
    # Basic check: point should be in front of both cameras
    X_hom = np.append(x_3, 1.0)

    X_left  = T_world_to_left  @ X_hom
    X_right = T_world_to_right @ X_hom
    if X_left[2] <= 0 or X_right[2] <= 0:
        return np.array([np.nan, np.nan, np.nan])

    # Distance check relative to IMU
    imu_pos = T_imu_to_world[:3, 3]
    dist = np.linalg.norm(x_3 - imu_pos)
    if dist > max_distance:
        return np.array([np.nan, np.nan, np.nan])

    return x_3

def initialize_landmarks(
    features,       # shape (4, n_features, n_timestamps)
    T_array,        # shape (n_timestamps, 4, 4)
    extL_T_imu,     # 4x4
    extR_T_imu,     # 4x4
    K_l, K_r,       
    max_distance=500
):
    """
    For each landmark, find its first valid observation and use stereo
    linear triangulation to initialize its 3D position in the world frame.
    Invalid or outlier landmarks are set to NaN.
    """
    n_features = features.shape[1]
    n_timestamps = features.shape[2]
    landmarks = np.full((n_features, 3), np.nan)

    success_count = 0
    for i in tqdm(range(n_features), desc="Initializing landmarks"):
        first_obs = None
        for t in range(n_timestamps):
            # Check if landmark i is visible at time t
            if not np.any(features[:, i, t] == -1):
                first_obs = t
                break
        if first_obs is not None:
            z = features[:, i, first_obs]  # (ul, vl, ur, vr)
            T_imu_world = T_array[first_obs]
            pos_3d = initialize_landmark_from_measurement(
                z, T_imu_world, extL_T_imu, extR_T_imu, K_l, K_r, max_distance=max_distance
            )
            landmarks[i] = pos_3d
            if not np.any(np.isnan(pos_3d)):
                success_count += 1

    print(f"Initialized {success_count}/{n_features} landmarks successfully.")
    return landmarks

if __name__ == '__main__':
    num = "02"
    file_name = f"/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR3/data/dataset{num}/dataset{num}.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(file_name)
    
    T_array = ekf_prediction(v_t, w_t, timestamps)  # (n_timestamps,4,4)
    landmarks = initialize_landmarks(features, T_array, extL_T_imu, extR_T_imu, K_l, K_r, max_distance=1000)
    
    fig, ax = visualize_trajectory_2d(T_array, path_name="IMU Trajectory", show_ori=False)
    ax.set_title(f"Dataset {num}: IMU Prediction & Landmarks")
    
    valid_idx = ~np.isnan(landmarks[:, 0])
    ax.scatter(landmarks[valid_idx, 0], landmarks[valid_idx, 1], c='b',s=5, marker='o', label='Landmarks')
    ax.legend()
    plt.show()
