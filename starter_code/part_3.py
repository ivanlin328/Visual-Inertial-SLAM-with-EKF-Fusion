import numpy as np
from pr3_utils import load_data,inversePose, projection,projectionJacobian,visualize_trajectory_2d
from part_1 import ekf_prediction
from tqdm import tqdm
import scipy.sparse as sp
import matplotlib.pyplot as plt

def  stereo_measurement_function(m_i,T_imu_to_world,extL_T_imu,extR_T_imu,K_l,K_r):
    """
    Project a 3D landmark m_i (in world frame) into stereo image coords [ul, vl, ur, vr].
    Using pr3_utils:
      - inversePose(T) to get T^{-1}
      - projection(ph) to divide by Z in homogeneous coords
    """
    #Transform from world -> IMU frame
    mi_to_hom = np.array([m_i[0], m_i[1], m_i[2], 1])                           
    T_world_to_imu = inversePose(T_imu_to_world[np.newaxis, ...])[0]   #np.newaxis... -> shape[1,4,4] [0]->shape[4,4]
    m_imu = T_world_to_imu @ mi_to_hom
    
    #left camera projection
    m_left = extL_T_imu @ m_imu
    m_left = m_left[np.newaxis,:]         # shape(1,4)
    #   projection => divides by z, returns [x/z, y/z, z/z, 1]
    left_projection= projection(m_left)   # shape (1,4)
    
    ul = K_l[0,0] * left_projection[0,0] + K_l[0,2]  # fu*(x/z) + cu
    vl = K_l[1,1] * left_projection[0,1] + K_l[1,2]  # fv*(y/z) + cv
    
    #right camera projection
    m_right = extR_T_imu @ m_imu
    m_right = m_right[np.newaxis,:]         # shape(1,4)
    #   projection => divides by z, returns [x/z, y/z, z/z, 1]
    right_projection= projection(m_right)   # shape (1,4)
    
    ur = K_r[0,0] * right_projection[0,0] + K_r[0,2]  # fu*(x/z) + cu
    vr = K_r[1,1] * right_projection[0,1] + K_r[1,2]  # fv*(y/z) + cv

    
    return np.array([ul, vl, ur, vr])

def stero_measurement_jacobian(m_i,T_imu_to_world,extL_T_imu,extR_T_imu,K_l,K_r):
    """
    Return:
      H_i: shape (4,3), Jacobian wrt. (x,y,z).
    """
    m_i_hom = np.array([m_i[0], m_i[1], m_i[2], 1.0])
    T_world_to_imu = inversePose(T_imu_to_world[np.newaxis, ...])[0] #(4x4)

    
    T_world_to_left = extL_T_imu @ T_world_to_imu
    T_world_to_right = extR_T_imu @ T_world_to_imu

    
    leftCamPoint = T_world_to_left @ m_i_hom
    leftCamPoint_1x4 = leftCamPoint[np.newaxis, :]
    J_proj_left = projectionJacobian(leftCamPoint_1x4)[0]  # (4,4)
    fx, fy = K_l[0, 0], K_l[1, 1]
    J_intrinsic_left = np.array([
        [fx, 0,  0,  0],
        [0,  fy, 0,  0]
    ], dtype=float)
    J_left_2x4 = J_intrinsic_left @ J_proj_left  # (2,4)
    R_left = T_world_to_left[:3, :3]
    J_left_2x3 = J_left_2x4[:, :3] @ R_left       # (2,3)


    rightCamPoint = T_world_to_right @ m_i_hom
    rightCamPoint_1x4 = rightCamPoint[np.newaxis, :]
    J_proj_right = projectionJacobian(rightCamPoint_1x4)[0]  # (4,4)
    fx_r, fy_r = K_r[0, 0], K_r[1, 1]
    J_intrinsic_right = np.array([
        [fx_r, 0,   0,  0],
        [0,   fy_r, 0,  0]
    ], dtype=float)
    J_right_2x4 = J_intrinsic_right @ J_proj_right  # (2,4)
    R_right = T_world_to_right[:3, :3]
    J_right_2x3 = J_right_2x4[:, :3] @ R_right       # (2,3)

    
    H_i = np.vstack([J_left_2x3, J_right_2x3])  # (4,3)
    return H_i
def initialize_landmark_from_measurement(
    z,                    # (ul, vl, ur, vr)
    T_imu_to_world,       # 4x4, IMU->World transform
    extL_T_imu,           # 4x4, IMU->LeftCam transform
    extR_T_imu,           # 4x4, IMU->RightCam transform
    K_l, K_r,             # 3x3 intrinsics (left, right)
    eps=1e-6,
    max_distance=200
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
# Modify your EKF_landmark_update function to return both initial and updated landmarks
def EKF_landmark_update(features, T_array, extL_T_imu, extR_T_imu, K_l, K_r,
                        Q_init=1e2, R_meas=1e1, reproj_error_thresh=50.0):
    M = features.shape[1]
    n_timestamps = features.shape[2]
    x = np.zeros((M, 3))
    initial_x = np.zeros((M, 3))  # Store initial triangulated positions
    P = np.array([np.eye(3) * Q_init for _ in range(M)])
    initialized = np.zeros(M, dtype=bool)
    R_mat = np.eye(4) * R_meas
    I3 = np.eye(3)

    for t in tqdm(range(n_timestamps)):  
        mask = np.all(features[:, :, t] != -1, axis=0)
        visible_ids = np.where(mask)[0]
        if len(visible_ids) == 0:
            continue

        for i in visible_ids:
            z_i = features[:, i, t]
            if not initialized[i]:
                m_init = initialize_landmark_from_measurement(z_i, T_array[t], extL_T_imu, extR_T_imu, K_l, K_r)
                x[i] = m_init
                initial_x[i] = m_init.copy()  
                initialized[i] = True
                continue

            m_i = x[i]
            z_pred = stereo_measurement_function(m_i, T_array[t], extL_T_imu, extR_T_imu, K_l, K_r)
            y = z_i - z_pred
            reproj_error = np.sum(y**2)

            if reproj_error > reproj_error_thresh:
                continue  

            H_i = stero_measurement_jacobian(m_i, T_array[t], extL_T_imu, extR_T_imu, K_l, K_r)
            S = H_i @ P[i] @ H_i.T + R_mat
            K_gain = P[i] @ H_i.T @ np.linalg.inv(S)
            x[i] = m_i + K_gain @ y
            P[i] = (I3 - K_gain @ H_i) @ P[i]

    return x.reshape(-1), initial_x.reshape(-1), P

# Then modify your main code to use both initial and updated landmarks
if __name__ == '__main__':
    num = "01"
    file_name = f"/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR3/data/dataset{num}/dataset{num}.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(file_name)
    T_array = ekf_prediction(v_t, w_t, timestamps)  # (n_timestamps, 4, 4)
    
    # Get both updated and initial landmarks
    X, X_initial, P = EKF_landmark_update(
        features, T_array, extL_T_imu, extR_T_imu, K_l, K_r,
        Q_init=1e2, R_meas=1e1,reproj_error_thresh=50.0             
    )
    
    # Reshape for easier processing
    landmarks = X.reshape(-1, 3)
    initial_landmarks = X_initial.reshape(-1, 3)

    imu_positions = np.array([T[:3, 3] for T in T_array])  # (n_timestamps, 3)


    distances_updated = np.min(np.linalg.norm(landmarks[:, np.newaxis, :] - imu_positions[np.newaxis, :, :], axis=2), axis=1)


    distances_initial = np.min(np.linalg.norm(initial_landmarks[:, np.newaxis, :] - imu_positions[np.newaxis, :, :], axis=2), axis=1)


    max_dist = 40
    max_dist_initial =60
    max_val = 200
    valid_idx_updated = (distances_updated < max_dist) & (~np.isnan(landmarks).any(axis=1)) & (np.abs(landmarks).max(axis=1) < max_val)
    valid_idx_initial = (distances_initial < max_dist_initial) & (~np.isnan(initial_landmarks).any(axis=1)) & (np.abs(initial_landmarks).max(axis=1) < max_val)


    landmarks_filtered = landmarks[valid_idx_updated]
    initial_landmarks_filtered = initial_landmarks[valid_idx_initial]


    fig, ax = visualize_trajectory_2d(np.array(T_array), path_name="EKF IMU Prediction", show_ori=False)
    ax.set_title(f"Dataset{num} EKF IMU Prediction with Landmarks")


    ax.scatter(initial_landmarks_filtered[:, 0], initial_landmarks_filtered[:, 1],
            marker='o', s=20, color='green', label='Initial Landmarks')


    ax.scatter(landmarks_filtered[:, 0], landmarks_filtered[:, 1],
            marker='o', s=5, color='blue', label='EKF Updated Landmarks')


    ax.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    

    