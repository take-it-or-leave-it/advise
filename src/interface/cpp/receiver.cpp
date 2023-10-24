#include "manager.h"

char tcp_server_ip[] = "192.168.203.190";

void ANTReceive(advise::InfoTable& recv_data){
    //printf("Receiver App is running.. \n");
    char recvmsg[1024];

    Listen_from_Ardu(recvmsg, DATASIZE_DILM);

    memcpy(&recv_data,recvmsg ,sizeof(recv_data));
    
}

void Listen_from_Ardu(char recvmsg[], int size){
    int cnt = 0;
    delay(1000);
    while(serialDataAvail(serial_port)){
        char get_data = serialGetchar (serial_port);
        //printf("get: %c\n", get_data);
        recvmsg[cnt++] = get_data;
        if(cnt == size){
            // Parse(recvmsg);
            //printf("received: %s\n", recvmsg);
            cnt = 0;
        }
    }
}

void TCP_Receive(char recvmsg[], char ip[]){
    int clnt_sock;
    tcp_port = 10000;
    
    struct sockaddr_in st_serv_addr;
    
    char sendmsg[] = "Test_clnt";
    
    // char ip[] = "192.168.203.190";
    
    clnt_sock = socket(PF_INET, SOCK_STREAM, 0);
    
    memset(&st_serv_addr,0,sizeof(st_serv_addr));
    st_serv_addr.sin_family = AF_INET;
    st_serv_addr.sin_addr.s_addr = inet_addr(ip);
    st_serv_addr.sin_port = htons(tcp_port);
    
    int connret = connect(clnt_sock,
                            (struct sockaddr*) &st_serv_addr,
                            sizeof(st_serv_addr));
                            
    printf("connection is successful!\n");
    
    write(clnt_sock, sendmsg, sizeof(sendmsg) );
    int readstrlen = recv(clnt_sock, recvmsg, 1024, 0);
    printf("received: %s\n", recvmsg);
    
    close(clnt_sock);
}