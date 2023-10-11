#include "advise/core0_tasks.h"


void Core0_5msTask(boost::mutex& global_mutex, const FrameData& frame_data_structure, const advise::InfoTable near_car_tables, advise::InfoTable& car_message_table){
    //mutex lock
    global_mutex.lock();

    //MeasureDistanceSide

    //Get front Rader distance from "Get Mavlink" node.(by topic)

    //check situation 

    //process situation

    //make car_message_table

    //mutext unlock
    global_mutex.unlock();

}