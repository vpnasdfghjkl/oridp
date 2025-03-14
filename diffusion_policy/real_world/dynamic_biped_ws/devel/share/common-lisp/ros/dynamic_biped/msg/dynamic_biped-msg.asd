
(cl:in-package :asdf)

(defsystem "dynamic_biped-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "ECJointMotordata" :depends-on ("_package_ECJointMotordata"))
    (:file "_package_ECJointMotordata" :depends-on ("_package"))
    (:file "QuaternionArray" :depends-on ("_package_QuaternionArray"))
    (:file "_package_QuaternionArray" :depends-on ("_package"))
    (:file "armHandPose" :depends-on ("_package_armHandPose"))
    (:file "_package_armHandPose" :depends-on ("_package"))
    (:file "handRotation" :depends-on ("_package_handRotation"))
    (:file "_package_handRotation" :depends-on ("_package"))
    (:file "handRotationEular" :depends-on ("_package_handRotationEular"))
    (:file "_package_handRotationEular" :depends-on ("_package"))
    (:file "recordArmHandPose" :depends-on ("_package_recordArmHandPose"))
    (:file "_package_recordArmHandPose" :depends-on ("_package"))
    (:file "robotArmInfo" :depends-on ("_package_robotArmInfo"))
    (:file "_package_robotArmInfo" :depends-on ("_package"))
    (:file "robotArmQVVD" :depends-on ("_package_robotArmQVVD"))
    (:file "_package_robotArmQVVD" :depends-on ("_package"))
    (:file "robotHandPosition" :depends-on ("_package_robotHandPosition"))
    (:file "_package_robotHandPosition" :depends-on ("_package"))
    (:file "robotHeadMotionData" :depends-on ("_package_robotHeadMotionData"))
    (:file "_package_robotHeadMotionData" :depends-on ("_package"))
    (:file "robotPhase" :depends-on ("_package_robotPhase"))
    (:file "_package_robotPhase" :depends-on ("_package"))
    (:file "robotQVTau" :depends-on ("_package_robotQVTau"))
    (:file "_package_robotQVTau" :depends-on ("_package"))
    (:file "robotTorsoState" :depends-on ("_package_robotTorsoState"))
    (:file "_package_robotTorsoState" :depends-on ("_package"))
    (:file "robot_hand_eff" :depends-on ("_package_robot_hand_eff"))
    (:file "_package_robot_hand_eff" :depends-on ("_package"))
    (:file "walkCommand" :depends-on ("_package_walkCommand"))
    (:file "_package_walkCommand" :depends-on ("_package"))
  ))