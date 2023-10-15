#ifndef CORE0_TASKS_H
#define CORE0_TASKS_H

#include <ros/ros.h>
#include <boost/thread.hpp>
#include "advise/applications.hpp"
#include "advise/InfoTable.h"

void Core0_5msTask(boost::mutex& global_mutex, const FrameData& frame_data_structure, const advise::InfoTable near_car_tables, advise::InfoTable& car_message_table);
void Core0_10msTask(boost::mutex& global_mutex, advise::InfoTable near_car_tables, const advise::InfoTable& car_message_table);
void Core0_30msTask(boost::mutex& global_mutex, FrameData& frame_data_structure);

#endif