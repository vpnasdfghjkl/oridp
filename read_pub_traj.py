import rospy
import numpy as np
import json
from kuavo_msgs.msg import sensorsData
from sensor_msgs.msg import JointState
import os
# 存储最近10组数据
history = []

# JSON 文件路径
script_dir = os.path.dirname(__file__)
json_file = os.path.join(script_dir, "joint_avg.json")

# 订阅回调函数
def callback(msg):
    global history
    
    # 取 joint_q[12:26] 关节角度（单位：弧度 -> 转换为角度）
    joint_angles = [np.rad2deg(float(i)) for i in msg.joint_data.joint_q[12:26]]
    
    # 添加到历史记录
    history.append(joint_angles)
    
    # 仅保留最近 10 组数据
    if len(history) > 10:
        history.pop(0)
    
    # 计算平均值
    avg_joint_angles = np.mean(history, axis=0).tolist()
    
    # 保存到 JSON
    with open(json_file, "w") as f:
        json.dump(avg_joint_angles, f)
    
    rospy.loginfo(f"Saved joint avg to {json_file}: {avg_joint_angles}")

# 读取 JSON 文件
def load_joint_from_json():
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# 发布关节角度
def publish_joint():
    pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
    rospy.sleep(1)  # 确保 Publisher 初始化完成
    
    # 读取存储的均值
    joint = load_joint_from_json()
    if joint is None:
        rospy.logwarn("No joint data found in JSON, skipping publish.")
        return
    
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = ["arm_joint_" + str(i) for i in range(1, 15)]
    msg.position = [i for i in joint]  # 转换回弧度
    rospy.loginfo(f"Published joint: {msg.position}")
    # rospy.sleep(5)  # 等待 10 秒
    pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("joint_listener")
    rospy.Subscriber("/sensors_data_raw", sensorsData, callback)
    # rospy.sleep(1)  # 等待 Subscriber 初始化完成
    # publish_joint()
    rospy.spin()
