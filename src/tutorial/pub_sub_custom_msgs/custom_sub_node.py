import rospy
from advise.msg import SimpleMsg

#include <sensor_msgs/NavSatFix.h>
#sensor_msgs::NavSatFix current_pose;

rospy.init_node("basic_sub_node")
rate = rospy.Rate(5.0)

def data_handler(raw_string):
    rospy.loginfo("Received! : "+raw_string.data)

data_receiver = rospy.Subscriber('advise/tutorial/hello',SimpleMsg,data_handler)

if __name__ == "__main__":
	while not rospy.is_shutdown():
		rate.sleep()
        
	