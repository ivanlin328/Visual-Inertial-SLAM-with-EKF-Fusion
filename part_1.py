from pr3_utils import load_data,axangle2pose, pose2adpose, visualize_trajectory_2d
import numpy as np
import matplotlib.pyplot as plt

def ekf_prediction(v_t,w_t,timestamps):
    num_steps = len(timestamps)
    T_array = np.zeros((num_steps, 4, 4))
    T = np.eye(4)
    T_array[0] = T
    
    for k in range(num_steps-1):
        dt= timestamps[k+1]-timestamps[k]
        #import pdb
        twist_i = np.hstack((v_t[k],w_t[k])) #shape [6,]
        #pdb.set_trace() 
        # deltaT = exp( hat(twist) * dt) )
        deltaT = axangle2pose(twist_i * dt)
        T = T @ deltaT
        T_array[k + 1] = T
    return T_array
    
    
if __name__ == '__main__':
    num="01"
    file_name = f"/Users/ivanlin328/Desktop/UCSD/Winter 2025/ECE 276A/ECE276A_PR3/data/dataset{num}/dataset{num}.npy"
    v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu= load_data(file_name)
    
    T_array = ekf_prediction(v_t, w_t, timestamps)
    fig, ax = visualize_trajectory_2d(np.array(T_array),path_name="Unknown",show_ori=False)
    ax.set_title(f"Dataset{num} EKF IMU Prediction Trajectory")
    plt.show()
