#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <boost/bind.hpp>
#include <stdio.h>
#include <string>
#include "advise/InfoTable.h"
#include "advise/Core0_Tasks.hpp"
#include "advise/applications.hpp"
using namespace ros;

advise::InfoTable near_car_tables[100]; 
advise::InfoTable car_message_table;
advise::FrameData frame_data_structure;

boost::mutex global_mutex;

void callback_5msTask(const TimerEvent& event){
   Core0_5msTask(global_mutex, frame_data_structure,near_car_tables,car_message_table);
}
void callback_10msTask(const TimerEvent& event){
   Core0_10msTask(global_mutex,near_car_tables,car_message_table );
}
void callback_30msTask(const TimerEvent& event){
   Core0_30msTask(global_mutex,frame_data_structure );
}
 
int main(int argc, char **argv)
{
   init(argc, argv, "main_app_node");
   
   NodeHandle n;
   Publisher requestCarDetection = n.advertise<std_msgs::String>("advise/request/cardetection");
   Publisher requestGPS = n.advertise<std_msgs::String>("advise/request/GPS");
   // Subscriber resultCarDetection = n.subscribe<mavros_msgs::State>("advise/result/cardetection", 10, boost::bind(&state_cb,_1,);
   // Subscriber resultCarDetection = n.subscribe<mavros_msgs::State>("advise/result/cardetection", 10, state_cb);

   Rate rate(1000.0);   
   Timer task_5ms_timer = n.createTimer(Duration(0.005),callback_5msTask);
   Timer task_10ms_timer = n.createTimer(Duration(0.01),callback_10msTask);
   Timer task_30ms_timer = n.createTimer(Duration(0.03),callback_30msTask);

   while(ok()){

      spinOnce();
      rate.sleep();

   }
   return 0;
}