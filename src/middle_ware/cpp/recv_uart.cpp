#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <serial/serial.h>
#include <stdio.h>
#include <string>
using namespace ros;

/*
Last update : 23-10-10 19:46 / Chanbyeongee
https://github.com/garyservin/serial-example/blob/master/src/serial_example_node.cpp

위 코드를 참고하였음

*/

serial::Serial ser;

int main()
{
    init("recv_uart_node");
    NodeHandle n;

    string uart_port;
    int baudrate;
    n.param<std::string>("uart_port", uart_port,"/dev/ttyACM0");
    n.param<std::string>("baudrate", uart_port,9600);

    Publisher broadCaster = n.advertise<std_msgs::String>("advise/recv_uart",1000); 

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
            broadCaster.publish(result);
        }
      rate.sleep();

   }
   return 0;
}