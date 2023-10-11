#ifndef CORE0_TASKS_H
#define CORE0_TASKS_H

#include <ros/ros.h>

ros::NodeHandle n;
ros::Publisher requestCarDetection = n.advertise<std_msgs::String>("advise/request/cardetection");
ros::Publisher requestGPS = n.advertise<std_msgs::String>("advise/request/GPS");
ros::Subscriber resultCarDetection = n.subscribe<mavros_msgs::State>("advise/result/cardetection", 10, state_cb);
ros::Subscriber resultCarDetection = n.subscribe<mavros_msgs::State>("advise/result/cardetection", 10, state_cb);

void Core0_5msTask(ROS::NodeHandle);
void Core0_10msTask(ROS::NodeHandle);
void Core0_30msTask(ROS::NodeHandle);

#endif