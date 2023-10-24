#include <RF24_config.h>
#include <RF24.h>
#include <printf.h>
#include <nRF24L01.h>
#include <SPI.h>

#define CE_PIN            7
#define CSN_PIN           8
#define RX_MODE           0
#define TX_MODE           1
#define PAYLOAD_SIZE      28
#define INIT_SUCCESS      1
#define INIT_FAILURE      0
#define TX_SUCCESS        1
#define TX_FAILURE        0

void Run();
void Transfer();
void Recieve();

RF24 radio(CE_PIN, CSN_PIN);

char payload[28] = "";
int pos = 0, res = 0;
int timeout = 1000, time_cnt = 0;
bool init_flag;
uint8_t address[][6] = { "1Node", "2Node" };

void setup() {
  Serial.begin(115200);
  while(!Serial) {}
  while(!radio.begin() && time_cnt++ < timeout){
    delay(10);
  }
  char input = Serial.parseInt();
  if(time_cnt == timeout){
    Serial.println(F("radio hardware is not responding!!"));
    init_flag = INIT_FAILURE;
  }
  else{
    pinMode(2, INPUT);
    pinMode(4, OUTPUT);
    digitalWrite(4, LOW);
    radio.setPALevel(RF24_PA_LOW);
    radio.setPayloadSize(PAYLOAD_SIZE);
    radio.openReadingPipe(0, address[0]);
    radio.openWritingPipe(address[0]);
    radio.startListening();
    init_flag = INIT_SUCCESS;
    // Serial.println("Init Success!!");
  } 
}  // setup

void loop(){
  if(init_flag == INIT_SUCCESS){
    Run();
  }
}

void Run(){
  if(digitalRead(2)){
    Transfer();
  }
  else Receive();
}

void Transfer(){
  radio.stopListening();
  time_cnt = 0;
  pos = 0;
  res = TX_FAILURE;
  char data;
  while(pos < PAYLOAD_SIZE && time_cnt < timeout){
    if(Serial.available()) {
      data = Serial.read();
    }
    else {
      time_cnt += 10;
      delay(1);
      continue;
    }
    payload[pos++] = data;
    // sprintf(payload + pos++, "%c", data);
    if(pos == PAYLOAD_SIZE){
      Serial.println(payload);
      time_cnt = 0;
      while(TX_FAILURE == (res = radio.write(&payload, sizeof(PAYLOAD_SIZE)))){
        delay(50);
        time_cnt += 100;
        if(time_cnt == timeout){
          Serial.println('F');
          time_cnt = 0;
          break;
        }
      }
      if(TX_SUCCESS == res) Serial.println('S');
      break;
    }
  }
  if(time_cnt == timeout) Serial.println("transfer failed");
  radio.startListening();
}

void Receive(){    
  uint8_t pipe;
  uint8_t bytes;
  
  // digitalWrite(4, HIGH);

  if(radio.available(&pipe)){
    bytes = radio.getPayloadSize();
    radio.read(&payload, bytes);
    for(int i=0; i<PAYLOAD_SIZE; i++)
      Serial.print(payload[i]);
    // Serial.println();
  }
  // digitalWrite(4, LOW);
  delay(5);
}
  
