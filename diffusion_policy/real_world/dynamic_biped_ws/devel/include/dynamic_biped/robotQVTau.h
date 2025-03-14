// Generated by gencpp from file dynamic_biped/robotQVTau.msg
// DO NOT EDIT!


#ifndef DYNAMIC_BIPED_MESSAGE_ROBOTQVTAU_H
#define DYNAMIC_BIPED_MESSAGE_ROBOTQVTAU_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace dynamic_biped
{
template <class ContainerAllocator>
struct robotQVTau_
{
  typedef robotQVTau_<ContainerAllocator> Type;

  robotQVTau_()
    : q()
    , v()
    , tau()  {
    }
  robotQVTau_(const ContainerAllocator& _alloc)
    : q(_alloc)
    , v(_alloc)
    , tau(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _q_type;
  _q_type q;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _v_type;
  _v_type v;

   typedef std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> _tau_type;
  _tau_type tau;





  typedef boost::shared_ptr< ::dynamic_biped::robotQVTau_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dynamic_biped::robotQVTau_<ContainerAllocator> const> ConstPtr;

}; // struct robotQVTau_

typedef ::dynamic_biped::robotQVTau_<std::allocator<void> > robotQVTau;

typedef boost::shared_ptr< ::dynamic_biped::robotQVTau > robotQVTauPtr;
typedef boost::shared_ptr< ::dynamic_biped::robotQVTau const> robotQVTauConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dynamic_biped::robotQVTau_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dynamic_biped::robotQVTau_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dynamic_biped::robotQVTau_<ContainerAllocator1> & lhs, const ::dynamic_biped::robotQVTau_<ContainerAllocator2> & rhs)
{
  return lhs.q == rhs.q &&
    lhs.v == rhs.v &&
    lhs.tau == rhs.tau;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dynamic_biped::robotQVTau_<ContainerAllocator1> & lhs, const ::dynamic_biped::robotQVTau_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dynamic_biped

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::robotQVTau_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::robotQVTau_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::robotQVTau_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b3aa74a32b604340b47572dd2a0b70d4";
  }

  static const char* value(const ::dynamic_biped::robotQVTau_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb3aa74a32b604340ULL;
  static const uint64_t static_value2 = 0xb47572dd2a0b70d4ULL;
};

template<class ContainerAllocator>
struct DataType< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dynamic_biped/robotQVTau";
  }

  static const char* value(const ::dynamic_biped::robotQVTau_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float64[] q\n"
"float64[] v\n"
"float64[] tau\n"
;
  }

  static const char* value(const ::dynamic_biped::robotQVTau_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.q);
      stream.next(m.v);
      stream.next(m.tau);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct robotQVTau_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dynamic_biped::robotQVTau_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dynamic_biped::robotQVTau_<ContainerAllocator>& v)
  {
    s << indent << "q[]" << std::endl;
    for (size_t i = 0; i < v.q.size(); ++i)
    {
      s << indent << "  q[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.q[i]);
    }
    s << indent << "v[]" << std::endl;
    for (size_t i = 0; i < v.v.size(); ++i)
    {
      s << indent << "  v[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.v[i]);
    }
    s << indent << "tau[]" << std::endl;
    for (size_t i = 0; i < v.tau.size(); ++i)
    {
      s << indent << "  tau[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.tau[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // DYNAMIC_BIPED_MESSAGE_ROBOTQVTAU_H
