#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <stdio.h>
#include <string>
#include "advise/Core0_Tasks.h"
using namespace ros;

void callback_5msTask(const TimerEvent& event){
   Core0_5msTask();
}
void callback_10msTask(const TimerEvent& event){
   Core0_10msTask();
}
void callback_30msTask(const TimerEvent& event){
   Core0_30msTask();
}

node -> 프로세스 
int main(int argc, char **argv)
{
   init(argc, argv, "main_app_node");
   NodeHandle n;

   Rate rate(1000.0);   
   Timer task_5ms_timer = n.createTimer(Duration(0.005),callback_5msTask);
   Timer task_10ms_timer = n.createTimer(Duration(0.01),callback_10msTask);
   Timer task_30ms_timer = n.createTimer(Duration(0.03),callback_30msTask);



   while(ok()){

      string data = readUART();

      spinOnce();
      rate.sleep();

   }
   return 0;
}