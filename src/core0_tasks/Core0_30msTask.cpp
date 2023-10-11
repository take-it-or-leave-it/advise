#include "advise/core0_tasks.h"


void Core0_30msTask(boost::mutex& global_mutex, FrameData& frame_data_structure){
    //mutex lock
    global_mutex.lock();

    //Get result of carmera detection from "FrameCarDetection" node.(by topic)

    //parsing result 

    //store parsed data into frame_data_structure

    //mutext unlock 
    global_mutex.unlock();
}