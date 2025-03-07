// Generated by gencpp from file dynamic_biped/robot_hand_eff.msg
// DO NOT EDIT!


#ifndef DYNAMIC_BIPED_MESSAGE_ROBOT_HAND_EFF_H
#define DYNAMIC_BIPED_MESSAGE_ROBOT_HAND_EFF_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace dynamic_biped
{
template <class ContainerAllocator>
struct robot_hand_eff_
{
  typedef robot_hand_eff_<ContainerAllocator> Type;

  robot_hand_eff_()
    : header()
    , data()  {
    }
  robot_hand_eff_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , data(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> _data_type;
  _data_type data;





  typedef boost::shared_ptr< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> const> ConstPtr;

}; // struct robot_hand_eff_

typedef ::dynamic_biped::robot_hand_eff_<std::allocator<void> > robot_hand_eff;

typedef boost::shared_ptr< ::dynamic_biped::robot_hand_eff > robot_hand_effPtr;
typedef boost::shared_ptr< ::dynamic_biped::robot_hand_eff const> robot_hand_effConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dynamic_biped::robot_hand_eff_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dynamic_biped::robot_hand_eff_<ContainerAllocator1> & lhs, const ::dynamic_biped::robot_hand_eff_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.data == rhs.data;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dynamic_biped::robot_hand_eff_<ContainerAllocator1> & lhs, const ::dynamic_biped::robot_hand_eff_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dynamic_biped

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a120344537a3b099cc9ec9957d4619fc";
  }

  static const char* value(const ::dynamic_biped::robot_hand_eff_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa120344537a3b099ULL;
  static const uint64_t static_value2 = 0xcc9ec9957d4619fcULL;
};

template<class ContainerAllocator>
struct DataType< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dynamic_biped/robot_hand_eff";
  }

  static const char* value(const ::dynamic_biped::robot_hand_eff_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n"
"float32[] data\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
;
  }

  static const char* value(const ::dynamic_biped::robot_hand_eff_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.data);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct robot_hand_eff_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dynamic_biped::robot_hand_eff_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dynamic_biped::robot_hand_eff_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "data[]" << std::endl;
    for (size_t i = 0; i < v.data.size(); ++i)
    {
      s << indent << "  data[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.data[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // DYNAMIC_BIPED_MESSAGE_ROBOT_HAND_EFF_H
