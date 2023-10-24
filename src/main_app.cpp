#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <stdio.h>
#include <string>
#include <vector>
#include "advise/InfoTable.h"
#include "advise/FrameCarMsg.h"
#include "advise/MavlinkMsg.h"
#include "advise/core0_tasks.h"

using namespace ros;

std::vector<advise::InfoTable> near_car_tables; 
std::vector<advise::InfoTable> car_message_table;
advise::InfoTable toPush;
advise::FrameCarMsg frame_data_structure;
advise::PlayAudioMsg to_play_audio;
serial::Serial tofSerials[6];

bool is_5msTask_work = false;
bool is_10msTask_work = false;

void callback_AsyncUART(const advise::InfoTable& resultUART){
   ros::Time current = ros::Time::now();
   std::vector<advise::InfoTable> new_near_tables;
   for(int i=0;i<near_car_tables.size();i++){
      if(resultUART.srcCarId==near_car_tables[i].srcCarId){
         new_near_tables.push_back(resultUART);
         break;
      }
      else if((current - near_car_tables[i].timeStamp)<=TIME_ERROR_THRESHOLD){
         new_near_tables.push_back(near_car_tables[i]);
      }
   }
   near_car_tables = new_near_tables;
}

void callback_Mavlink(const advise::MavlinkMsg& resultMavlink){
   Core0_5msTask(toPush,to_play_audio,resultMavlink, tofSerials, frame_data_structure,near_car_tables,car_message_table);
   is_5msTask_work = true;
}
void callback_nearCar(const advise::InfoTable& nearCar){
   Core0_10msTask(nearCar, near_car_tables);
   bool is_10msTask_work = true;
}
void callback_FrameCarDetection(const advise::FrameCarMsg& resultCar){
   Core0_60msTask(resultCar,frame_data_structure);
}
 
int main(int argc, char **argv)
{
   
   init(argc, argv, "main_app_node");
   NodeHandle nh;
   //wiringPiSetup();
   // pinMode(11, INPUT);
   // pinMode(15, INPUT);

   Publisher requestCarDetection = nh.advertise<std_msgs::String>("advise/request/cardetection",10);
   //Publisher requestGPS = nh.advertise<std_msgs::String>("advise/request/GPS",10);
   Publisher requestPlayAudio = nh.advertise<advise::PlayAudioMsg>("advise/request/play_audio",10);
   Publisher requestPushUART = nh.advertise<advise::InfoTable>("advise/request/push_uart",10);
   Publisher requestPushMQTT = nh.advertise<advise::InfoTable>("advise/request/push_mqtt",10);

   Subscriber resultCarDetection = nh.subscribe("advise/frame_car", 10, callback_FrameCarDetection);
   Subscriber resultMavlink = nh.subscribe("advise/mavlink", 10, callback_Mavlink);
   Subscriber resultUART = nh.subscribe("advise/recv_uart", 10, callback_AsyncUART);
   Subscriber resultNearCar = nh.subscribe("advise/near_car", 10, callback_nearCar);

   Rate rate(1000.0);   
   //Timer task_5ms_timer = nh.createTimer(Duration(0.005),callback_5msTask);
   //Timer task_10ms_timer = nh.createTimer(Duration(0.01),callback_10msTask);
   //Timer task_30ms_timer = nh.createTimer(Duration(0.03),callback_30msTask);

   while(ok()){

      if(is_5msTask_work){
         requestPlayAudio.publish(to_play_audio);
         is_5msTask_work = false;
      }
      if(is_10msTask_work){
         for(auto elem: car_message_table){
            requestPushUART.publish(elem);
         }
         requestPushMQTT.publish(toPush);
         is_10msTask_work = false;
      }
      
      spinOnce();
      rate.sleep();

   }
   return 0;
}