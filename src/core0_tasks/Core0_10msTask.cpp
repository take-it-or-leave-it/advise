#include "advise/core0_tasks.h"


void Core0_10msTask(const advise::InfoTable& nearCar, std::vector<advise::InfoTable>& near_car_tables){

    ros::Time current = ros::Time::now();
    std::vector<advise::InfoTable> new_near_tables;
    for(int i=0;i<near_car_tables.size();i++){
        if(nearCar.srcCarId==near_car_tables[i].srcCarId){
            new_near_tables.push_back(nearCar);
            break;
        }
        else if(current - near_car_tables[i].timeStamp<=TIME_ERROR_THRESHOLD){
            new_near_tables.push_back(near_car_tables[i]);
        }
    }
    near_car_tables = new_near_tables;
}