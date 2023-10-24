#include "advise/interfaces.h"
#include <serial/serial.h>

float  MeasureDistanceSide(const serial::Serial serials[],float MultiTofData[]){
    int dist; 
    int check; //save check value
    int i;
    int uart[9]; //save data measured by LiDAR
    const int HEADER=0x59; //frame header of data package

    for(int n=0;n<6;n++){
        if (serials[n].read() == HEADER) { //assess data package frame header 0x59
            uart[0] = HEADER;
            if (serials[n].read() == HEADER) { //assess data package frame header 0x59
                uart[1] = HEADER;
                for (i = 2; i < 9; i++) { //save data in array
                    uart[i] = serials[n].read();
                }
            
                check = uart[0] + uart[1] + uart[2] + uart[3] + uart[4] + uart[5] + uart[6] + uart[7];
                if (uart[8] == (check & 0xff)){ 
                    dist = uart[2] + uart[3] * 256; 
                    MultiTofData[n]= dist;
                }
            }
        }
    }

}