"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import imageio
import numpy as np
import torch

import time
import rospy
import threading
import signal
import cv2
import sys

'''
zsh
conda_active
conda activate lerobot
source ~/catkin_ws_data_pilot/devel/setup.bash
rosservice call /arm_traj_change_mode "control_mode: 2"
unset PYTHONPATH
source ~/catkin_ws_data_pilot/devel/setup.bash

'''




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
if not torch.cuda.is_available():
     print(f"⚠️⚠️⚠️ running on cpu")
else:
    print(f">-< running on cuda")
# exit(0)
# from real_env_kuavo import KuavoEnv
from LeRobot_KuavoEnv import KuavoEnv
rospy.init_node("LeRobot")
task_name = "Task10-SuZhou"
with KuavoEnv(
        frequency=10,
        n_obs_steps=2,
        video_capture_fps=30,
        robot_publish_rate=500,

        img_buffer_size=30,
        robot_state_buffer_size=500,

        video_capture_resolution=(640, 480),

        
        ) as env:
        print("waiting for the obs buffer to be ready ......")
        
        env.obs_buffer.wait_buffer_ready(just_img=True)
        # time.sleep(5)
        obs,_,_,_,_ = env.get_obs(just_img=True)
        
        
        for k, v in obs.items():
            print(f'{k=}, \t {v.shape=}')
        while 1:
            obs,_,_,_,_ = env.get_obs(just_img=True)
            start_point = time.time()
            
            
            # =============
            # show sense img
            # =============
            imgs = []
            
            for k, v in obs.items():
                if "img01" in k:
                    img = obs[k][-1]
                    img = cv2.resize(img, (640, 480))
                    img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    imgs.append(img01_bgr)
            concatenated_img = np.hstack(imgs)  # 横向拼接图像
            cv2.imwrite('/home/leju-ali/hx/oridp/check1.png', concatenated_img)
            time.sleep(1)