#include "advise/core0_tasks.h"

void Core0_60msTask(const advise::FrameCarMsg& frameMsg, advise::FrameCarMsg& frame_data_structure){
    
    //Get result of carmera detection from "/advise/frame_car".(by topic)
    //ROS_INFO("THIS IS 60ms TASK\n");

    //parsing result 
    frame_data_structure.frontCar = frameMsg.frontCar;
    frame_data_structure.rearCar = frameMsg.rearCar;
    frame_data_structure.rearDistance = frameMsg.rearDistance;



}