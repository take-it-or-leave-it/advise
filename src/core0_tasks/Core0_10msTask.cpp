#include "advise/core0_tasks.h"


void Core0_10msTask(boost::mutex& global_mutex, advise::InfoTable near_car_tables, const advise::InfoTable& car_message_table){
    //mutex lock
    global_mutex.lock();

    //Get Near Car Table data from "Get Mavlink" node.(by topic)

    //make near car tables

    //make uart data from car_message_table

    //push uart data by calling "PUSH DATA TABLE UART"

    //mutext unlock 
    global_mutex.unlock();

}