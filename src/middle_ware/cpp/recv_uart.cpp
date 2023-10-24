#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <serial/serial.h>
#include <stdio.h>
#include <string>
#include "advise/InfoTable.h"
using namespace ros;

/*
Last update : 23-10-10 19:46 / Chanbyeongee
https://github.com/garyservin/serial-example/blob/master/src/serial_example_node.cpp

위 코드를 참고하였음

*/
serial::Serial ser;
string toSend;
void pushUART(const advise::InfoTable& data){
    toSend = "";
    toSend = bytes(data);
}

int main(int argc, char **argv)
{
    init(argc, argv,"recv_uart_node");
    NodeHandle n;

    std::string uart_port;
    int baudrate;
    n.param<std::string>("uart_port", uart_port,"/dev/ttyACM0");
    n.param<int>("baudrate", baudrate,9600);

    Publisher broadCaster = n.advertise<std_msgs::String>("advise/recv_uart",1000); 
    Subscriber pushUART = nh.subscribe<advise::InfoTable>("advise/request/push_uart", 10, callback_pushUART);
   try
    {
        ser.setPort(uart_port);
        ser.setBaudrate(baudrate);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(to);
        ser.open();
    }
    catch (serial::IOException& e)
    {
        ROS_ERROR_STREAM("Unable to open port ");
        return -1;
    }
    if(ser.isOpen()){
        ROS_INFO_STREAM("Serial Port initialized");
    }else{
        return -1;
    }

    Rate rate(5);

   while(ok()){

      spinOnce();
      if(ser.available()){
            
            std_msgs::String result;
            result.data = ser.read(ser.available());
            ser.write(toSend);
            broadCaster.publish(result);
        }
      rate.sleep();

   }
   return 0;
}