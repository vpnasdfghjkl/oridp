;; Auto-generated. Do not edit!


(when (boundp 'dynamic_biped::robot_hand_eff)
  (if (not (find-package "DYNAMIC_BIPED"))
    (make-package "DYNAMIC_BIPED"))
  (shadow 'robot_hand_eff (find-package "DYNAMIC_BIPED")))
(unless (find-package "DYNAMIC_BIPED::ROBOT_HAND_EFF")
  (make-package "DYNAMIC_BIPED::ROBOT_HAND_EFF"))

(in-package "ROS")
;;//! \htmlinclude robot_hand_eff.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass dynamic_biped::robot_hand_eff
  :super ros::object
  :slots (_header _data ))

(defmethod dynamic_biped::robot_hand_eff
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _data __data)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32[] _data
    (* 4    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get dynamic_biped::robot_hand_eff :md5sum-) "a120344537a3b099cc9ec9957d4619fc")
(setf (get dynamic_biped::robot_hand_eff :datatype-) "dynamic_biped/robot_hand_eff")
(setf (get dynamic_biped::robot_hand_eff :definition-)
      "Header header
float32[] data
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :dynamic_biped/robot_hand_eff "a120344537a3b099cc9ec9957d4619fc")


