#include "advise/core0_tasks.h"
#include <cmath>
#include <algorithm>


void Core0_5msTask( advise::InfoTable& toPush,advise::PlayAudioMsg& play_msg,const advise::MavlinkMsg& resultMavlink , const serial::Serial TOFserials[], const advise::FrameCarMsg& frame_data_structure, const std::vector<advise::InfoTable>& near_car_tables, std::vector<advise::InfoTable>& car_message_table){

    //MeasureDistanceSide

    float MultiTofData[6];
    
    //MeasureDistanceSide(TOFserials,MultiTofData);

    //check situation 
    bool is_1stSituation=false;
    bool is_2ndSituation=false;
    bool is_3rdSituation=false;
    bool is_receiver = false; //신호를 받았는지, 줄것인지, true가 신호를 받은경우
    bool is_safe = false; //true -> 그대로 진행, false -> 양보해야함
    
    /*
        상황 처리 알고리즘 완성하기
    */
    bool my_left_siganl=0;
    bool my_right_siganl=1;
    // bool my_left_siganl=(digitalRead(11)<CDS_LIMIT);
    // bool my_right_siganl=(digitalRead(15)<CDS_LIMIT);
    
    //신호를 킨 경우, 딜레마 상황인지 판단하기(둘다 킨 경우는 비상등이므로 처리 X)
    if(my_left_siganl ^ my_right_siganl){
        //내가 1번 상황에 처해있는지 판단하기
        //1번상황은 다른 차에서 깜빡이를 킨것과는 전혀 상관없음
        if(!frame_data_structure.frontCar){
            if(resultMavlink.frontDistance<=FRONT_LIMIT){
                is_1stSituation=true;
                is_receiver = false;
            }
        }
    }
    
    for(auto elem:near_car_tables){
        //주변 차량 정보로 부터 딜레마 상황 판단하기
        //해당 차량의 왼쪽 TOF값과 오른쪽 TOF값이 우리 차량의 반대의 값과 같은가
        //해당 차량의 정면 센서 값이 우리 차량의 뒷 센서 값과 같은가
        //딜레마 상황을 알려주는 쪽이 위험한지 아닌지도 알려줌
        float tof_arr[6]={elem.tof0,elem.tof1,elem.tof2,elem.tof3,elem.tof4,elem.tof5};
        advise::InfoTable to_send_table;

        to_send_table.recvCarId = elem.srcCarId;
        to_send_table.srcCarId = CAR_ID;
        to_send_table.lat = resultMavlink.lat;
        to_send_table.lon = resultMavlink.lon;
        to_send_table.vel = resultMavlink.vel;
        to_send_table.is_signal_on = my_left_siganl ^ my_right_siganl;
        to_send_table.signal_direction = my_left_siganl+my_right_siganl*2;
        to_send_table.tof0 = MultiTofData[0];
        to_send_table.tof1 = MultiTofData[1];
        to_send_table.tof2 = MultiTofData[2];
        to_send_table.tof3 = MultiTofData[3];
        to_send_table.tof4 = MultiTofData[4];
        to_send_table.tof5 = MultiTofData[5];

        toPush = to_send_table;
        toPush.recvCarId = -1;

        float SD = (elem.vel*0.5)+(elem.vel*0.27778)*(elem.vel*0.27778)*(23.535);//안전거리 구하기
        if(is_1stSituation){
            //1번 딜레마 상황이라, 상대방 차량과의 위치를 확인해야 한다.
             if(my_left_siganl){
                //상대차량과 내 정보가 같은지 확인함
                bool is_match = false;
                int match_idx =0;

                for(int n=0;n<5;n++){
                    bool is_parrel=true;
                    for(int t=std::max(0,n-2);t<3;t++){
                        if(std::abs(MultiTofData[t]-tof_arr[5-n+t])>TOF_ERROR_THRESHOLD){
                            is_parrel=false;
                            break;
                        }
                    }
                    if(is_parrel){
                        is_match=true;
                        match_idx=n;
                        break;
                    }
                }
                if(is_match){
            
                    // 같으면 1번 이 차량이 정보 전달해야함
                    to_send_table.dilema_id=1;
                    //거리에 따라서 변별하기
                    if(match_idx<=3){
                        //이미 가까이 있음
                        to_send_table.is_source_car_safe = false;
                        to_send_table.is_dest_car_safe = true;
                        is_safe=false;
                    }else if(match_idx==4){
                        
                        if(MultiTofData[2]> SD && tof_arr[3]>SD){
                            //안전거리 넘어에 있음 -> 양보해야함
                            to_send_table.is_source_car_safe = true;
                            to_send_table.is_dest_car_safe = false;
                            is_safe=true;
                        }else{
                            to_send_table.is_source_car_safe = false;
                            to_send_table.is_dest_car_safe = true;
                            is_safe=false;
                        }
                    }
                }
            }else{
                //오른쪽 깜빡이
                bool is_match = false;
                int match_idx =0;

                for(int n=0;n<5;n++){
                    bool is_parrel=true;
                    for(int t=std::max(0,n-2);t<3;t++){
                        if(std::abs(tof_arr[t]-MultiTofData[5-n+t])>TOF_ERROR_THRESHOLD){
                            is_parrel=false;
                            break;
                        }
                    }
                    if(is_parrel){
                        is_match=true;
                        match_idx=n;
                        break;
                    }
                }
                if(is_match){
                    // 같으면 1번 이 차량이 정보 전달해야함
                    to_send_table.dilema_id=1;
                    //거리에 따라서 변별하기
                    if(match_idx>=1){
                        //이미 가까이 있음
                        to_send_table.is_source_car_safe = false;
                        to_send_table.is_dest_car_safe = true;
                        is_safe=false;
                    }else if(match_idx==0){
                        
                        if(MultiTofData[2]> SD && tof_arr[3]>SD){
                            //안전거리 넘어에 있음 -> 양보해야함
                            to_send_table.is_source_car_safe = true;
                            to_send_table.is_dest_car_safe = false;
                            is_safe=true;
                        }else{
                            to_send_table.is_source_car_safe = false;
                            to_send_table.is_dest_car_safe = true;
                            is_safe=false;
                        }
                    }
                }
            }
        }
        else if(elem.dilema_id!=0){
           
            is_receiver=true;
            if(elem.dilema_id==1){
                //상대방 차량이 먼저 위험함
                is_1stSituation=true;
                if(elem.is_source_car_safe && !elem.is_dest_car_safe){
                    is_safe = false;
                }
                else if(!elem.is_source_car_safe && elem.is_dest_car_safe){
                    is_safe = true;
                }
                else is_safe = false;

            }else if(elem.dilema_id==2){
                //상대방 차량이 우선순위가 밀림
                is_2ndSituation=true;
                if(elem.is_source_car_safe && !elem.is_dest_car_safe){
                    is_safe = false;
                }
                else if(!elem.is_source_car_safe && elem.is_dest_car_safe){
                    is_safe = true;
                }
                else is_safe = false;

            }else if(elem.dilema_id==3){
                //상대방 차량이 우선순위가 밀림
                is_3rdSituation=true;
                if(elem.is_source_car_safe && !elem.is_dest_car_safe){
                    is_safe = false;
                }
                else if(!elem.is_source_car_safe && elem.is_dest_car_safe){
                    is_safe = true;
                }
                else is_safe = false;

            }
        }else{
            //2번 상황은 
            if(elem.is_signal_on){
                //상대방이 켰는데, 
                if(to_send_table.is_signal_on) {
                    //나도 켰음 -> 이경우는 상대방이 먼저임
                    if(3-to_send_table.signal_direction == elem.signal_direction){
                        //서로 같은 차로로 이동하는 경우 -> 2번 상황 체크하기
                        if(my_left_siganl){
                            //상대차량과 내 정보가 같은지 확인함
                            bool is_match = false;
                            
                            float line_distance = 0;

                            for(int n=0;n<5;n++){
                                bool is_parrel=true;
                                for(int t=std::max(0,n-2);t<3;t++){
                                    if(std::abs(MultiTofData[t]-tof_arr[5-n+t])>TOF_ERROR_THRESHOLD){
                                        is_parrel=false;
                                        break;
                                    }
                                    line_distance = tof_arr[5-n+t];
                                }
                                if(is_parrel){
                                    is_match=true;
                                    
                                    break;
                                }
                            }
                            if(is_match){
                                // 같으면 이제 거리 판별해야함
                                if(line_distance>=TOF_MIN_LIMIT && line_distance <= TOF_MAX_LIMIT){
                                    //2차선 넘어로 떨어져 있음 -> 딜레마 상황임
                                    is_2ndSituation = true;
                                    is_receiver = false;
                                    is_safe = false;
                                    to_send_table.dilema_id=2;
                                    to_send_table.is_source_car_safe=false;
                                    to_send_table.is_dest_car_safe=true;
                                }
                            }
                        }else{
                            //상대차량과 내 정보가 같은지 확인함
                            bool is_match = false;
                            
                            float line_distance = 0;

                            for(int n=0;n<5;n++){
                                bool is_parrel=true;
                                for(int t=std::max(0,n-2);t<3;t++){
                                    if(std::abs(tof_arr[t]-MultiTofData[5-n+t])>TOF_ERROR_THRESHOLD){
                                        is_parrel=false;
                                        break;
                                    }
                                    line_distance = tof_arr[t];
                                }
                                if(is_parrel){
                                    is_match=true;
                                    
                                    break;
                                }
                            }
                            if(is_match){
                                // 같으면 이제 거리 판별해야함
                                if(line_distance>=TOF_MIN_LIMIT && line_distance <= TOF_MAX_LIMIT){
                                    //2차선 넘어로 떨어져 있음 -> 딜레마 상황임
                                    is_2ndSituation = true;
                                    is_receiver = false;
                                    is_safe = false;
                                    to_send_table.dilema_id=2;
                                    to_send_table.is_source_car_safe=false;
                                    to_send_table.is_dest_car_safe=true;
                                }
                            }
                        }
                    }
                    else if(to_send_table.signal_direction == elem.signal_direction){
                        //3번 상황인지?
                        if(std::abs(elem.front_distance-frame_data_structure.rearDistance)<=TOF_ERROR_THRESHOLD){
                            to_send_table.dilema_id=3;
                            is_3rdSituation = true;
                            is_receiver = false;
                            
                            if(elem.front_distance>=SD){
                                to_send_table.is_source_car_safe=true;
                                to_send_table.is_dest_car_safe=false;
                                is_safe =true;
                            }
                            else{
                                to_send_table.is_source_car_safe=false;
                                to_send_table.is_dest_car_safe=true;
                                is_safe =false;
                            }
                        }
                    }
                }
            }
        }
        car_message_table.push_back(to_send_table);
    }

    play_msg.dilema_id = (is_1stSituation) + (is_2ndSituation*2)+ (is_3rdSituation*3);
    play_msg.is_receiver = is_receiver;
    play_msg.is_safe = is_safe;
    
}