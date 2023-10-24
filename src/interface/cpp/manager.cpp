#include "advise/manager.h"

void Transform_norm(const advise::InfoTable data){
    tx_buf.push(data);
}

ANTData_t Parse_norm(){

    advise::InfoTable rx_data = rx_buf.front();
    rx_buf.pop();

    return rx_data;
}

int SerialComm_Init(){
    if ((serial_port = serialOpen ("/dev/ttyACM0", 115200)) < 0)	
    {
        printf("Unable to open serial device\n");
        return 1 ;
    }

    if (wiringPiSetupGpio () == -1)			
    {
        printf("Unable to start wiringPi\n");
        return 1 ;
    }

    pinMode(6, OUTPUT);
    digitalWrite(6, LOW);

    return 0;
}

