#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <stdio.h>
#include <string>

//DTO for publish
std_msgs::String toSendData;


int main(int argc, char **argv)
{
   ros::init(argc, argv, "pub_setpoints");
   ros::NodeHandle n;

    //publishers
    ros::Publisher advertiseData = n.advertise<std_msgs::String>("advise/tutorial/hello",10);
   
//   //subscribers
//   //ros::Subscriber target_sub = n.subscribe<geographic_msgs::GeoPoseStamped>("targeting",10,target_cb); //Realed to SERVER
//   ros::Subscriber state_sub = n.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
//   ros::Subscriber global_sub = n.subscribe<sensor_msgs::NavSatFix>("mavros/global_position/global", 10, global_cb);     
    
   ros::Rate rate(5.0);
   
    
   ros::Time last_request = ros::Time::now();

    int cnt = 0;

   while(ros::ok()){

        toSendData.data = "Hello" + std::to_string(cnt);
        advertiseData.publish(toSendData);
    
        ROS_INFO("Sent Data : %s",toSendData.data.c_str());
        
        cnt++;

        ros::spinOnce();
        rate.sleep();

   }
   return 0;
}