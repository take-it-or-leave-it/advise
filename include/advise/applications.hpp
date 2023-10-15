#ifndef APPLICATIONS_H
#define APPLICATIONS_H

typedef struct {
    bool is_front_car;
    bool is_rear_car;
    float distance_rear_car;
}FrameData;

void CheckSituation();
void GetUartData();

#endif