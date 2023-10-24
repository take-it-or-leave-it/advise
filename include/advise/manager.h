#ifndef MANAGER_H
#define MANAGER_H

#include <queue>
#include "advise/InfoTable.h"
using namespace std;

queue<advise::InfoTable> tx_buf, rx_buf;
//queue<advise::InfoTable> dilemma_tx_buf, dilemma_rx_buf;

//void Init_Manager();
void SerialComm_Init();

void ANTReceive();
void ANTTransfer();

void Transform_norm(const ANTData_t data);
//void Transform_dilm(const Dilemma_t data);
ANTData_t Parse_norm();
Dilemma_t Parse_dilm();

void Put_to_Ardu(char sendmsg[], int size);
void TCP_Transfer(char sendmsg[], int size);

void Listen_from_Ardu(char recvmsg[], int size);
void TCP_Receive(char recvmsg[], char ip[]);
void Parse(const char recvmsg[]);

int serial_port;
int tcp_port;

#endif