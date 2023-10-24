#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <serial/serial.h>
#include <stdio.h>
#include <string>
#include "advise/InfoTable.h"
#include "advise/manager.h"
using namespace ros;

serial::Serial ser;
string toSend;
void pushUART(const advise::InfoTable& data){
    ANTTransfer(data);
}

int main(int argc, char **argv)
{
    init(argc, argv,"recv_uart_node");
    NodeHandle n;
    SerialComm_Init();

    advise::InfoTable rx_data;
    advise::InfoTable tx_data;
    //ANTData_t a_data;
    //Dilemma_t d_data;

    n.param<std::string>("uart_port", uart_port,"/dev/ttyACM0");
    n.param<int>("baudrate", baudrate,9600);

    Publisher broadCaster = n.advertise<std_msgs::InfoTable>("advise/recv_uart",1000); 
    Subscriber pushUART = nh.subscribe<advise::InfoTable>("advise/request/push_uart", 10, callback_pushUART);
   
    Rate rate(5);

    while(ok()){
        ANTReceive(rx_data);
        broadCaster.publish(rx_data);
        
        rate.sleep();
        spinOnce();
    }

   return 0;
}