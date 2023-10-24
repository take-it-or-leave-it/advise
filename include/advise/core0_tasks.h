#ifndef CORE0_TASKS_H
#define CORE0_TASKS_H

#include <ros/ros.h>
#include <vector>
//#include <wiringPi.h>
#include "advise/interfaces.h"
#include "advise/InfoTable.h"
#include "advise/FrameCarMsg.h"
#include "advise/MavlinkMsg.h"
#include "advise/PlayAudioMsg.h"

#define TOF_MIN_LIMIT 412.5
#define TOF_MAX_LIMIT 762.5 
#define TOF_ERROR_THRESHOLD 30.0
#define FRONT_LIMIT 50.0
#define CDS_LIMIT 200
#define CAR_ID 1

const ros::Duration TIME_ERROR_THRESHOLD =ros::Duration(10000);


void Core0_5msTask( advise::InfoTable& toPush,advise::PlayAudioMsg& play_msg,const advise::MavlinkMsg& resultMavlink , const serial::Serial TOFserials[], const advise::FrameCarMsg& frame_data_structure, const std::vector<advise::InfoTable>& near_car_tables, std::vector<advise::InfoTable>& car_message_table);
void Core0_10msTask( const advise::InfoTable& nearCar, std::vector<advise::InfoTable>& near_car_tables);
void Core0_60msTask(const advise::FrameCarMsg& frameMsg, advise::FrameCarMsg& frame_data_structure);

#endif