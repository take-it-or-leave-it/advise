#include "manager.h"

void ANTTransfer(const advise::InfoTable tx_data){
    //printf("Transmitter App is running.. \n");

    Put_to_Ardu(tx_data);
}

void Put_to_Ardu(dvise::InfoTable tx_data){
    int cnt = 0;
    char sendmsg = new char[sizeof(tx_data)];
    memcpy(sendmsg,&tx_data,sizeof(tx_data)); 
    
    digitalWrite(6, HIGH);
    while(cnt < size){
        serialPutchar(serial_port, sendmsg[cnt++]);
    }
    digitalWrite(6, LOW);
}

void TCP_Transfer(char sendmsg[], int size){
    int serv_sock, clnt_sock;
    tcp_port = 10000;
    
    // sockaddr_in 구조체 변수 선언
    struct sockaddr_in st_serv_addr;
    struct sockaddr_in st_clnt_addr;
    
    // 보내고 받을 버퍼 정의
    char recvmsg[1024];

    int recv_id = 1;
    double recv_lat = 10.0, recv_long = 4.0, recv_velo = 5.0;
    memcpy(sendmsg, &recv_id, sizeof(int));
    memcpy(sendmsg + 4, &recv_lat, sizeof(double));
    memcpy(sendmsg + 12, &recv_long, sizeof(double));
    memcpy(sendmsg + 20, &recv_velo, sizeof(double));
    
    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    
    memset(&st_serv_addr,0,sizeof(st_serv_addr));
    st_serv_addr.sin_family = AF_INET;
    st_serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    st_serv_addr.sin_port = htons(tcp_port);
    
    int bindret = bind(serv_sock, (struct sockaddr*) &st_serv_addr, sizeof(st_serv_addr) );
    
    printf("listen ..\n");
    int listenret = listen(serv_sock,10);
    
    int clnt_addr_size = sizeof(st_clnt_addr);
    clnt_sock = accept(serv_sock,
                            (struct sockaddr*) &st_clnt_addr,
                            &clnt_addr_size );
    printf("accepted client!\n");
    
    int readstrlen = recv(clnt_sock, recvmsg, sizeof(recvmsg)-1, 0);
    write(clnt_sock, sendmsg, size);
    printf("recv msg: %s\n", recvmsg);
    
    close(clnt_sock);
    close(serv_sock);
}