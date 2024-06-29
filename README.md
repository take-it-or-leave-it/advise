# 양자택일 : V2V 차량 딜레마존 사고방지 시스템
>
> Start of the project : 2023.06 <br>
>**제21회 ESW(임베디드 소프트웨어 경진대회) 현대자동차 공모부문 결선 진출작**
>
> Team leader : 서울과기대 ***박용석***<br>
> Team member : 경희대학교  ***이병찬, 권강환, 박광명***<br> 
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;대진대학교  ***장승민***<br>

## Video Link
[![Video Label](http://img.youtube.com/vi/4fHBYZ_cw4c/0.jpg)](https://youtu.be/4fHBYZ_cw4c)


## Objective
Development of an Accident Prevention System for Vehicle Dilemma Zones Using V2V Communication

**1. Resolving Dilemma Zones at Highway Entrance Ramps**<br>
&nbsp;&nbsp;&nbsp;1.1 Send safe entry signals to vehicles attempting to merge<br>
&nbsp;&nbsp;&nbsp;1.2 Send driving adjustment signals to surrounding vehicles<br>

**2. Resolving Dilemma Zones During Lane Changes Between Adjacent Lanes**<br>
&nbsp;&nbsp;&nbsp;2.1 Send lane change signals to priority vehicles<br>
&nbsp;&nbsp;&nbsp;2.2 Send lane change prohibition signals to secondary vehicles<br>

**3. Resolving Dilemma Zones During Lane Changes Within the Same Lane**<br>
&nbsp;&nbsp;&nbsp;3.1 Send lane change signals to priority vehicles<br>
&nbsp;&nbsp;&nbsp;3.2 Send lane change prohibition signals to secondary vehicles<br>


## Solution Approach

### 1. Vehicle Equipment Installation
- LiDAR
- RADAR
- Raspberry Pi
- Camera
- RF Antenna

### 2. Dilemma Zone Detection via Sensors
- Camera:
  - Use image processing techniques (MobileNetv2, OpenCV) to detect vehicles in the same lane
  - Employ stereo distance measurement technology to calculate inter-vehicle distances
- LiDAR:
  - Measure distances to vehicles in adjacent lanes
  - Detect the range of lanes in which vehicles are located

### 3. Risk Assessment and Signal Transmission
- Recognize the intended actions of the current vehicle
- When detecting safety distance risks or other dilemma zone hazards:
  - Send warning signals to the operating vehicle
  - Transmit appropriate signals to other vehicles within the dilemma zone

### 4. Real-time Processing and Communication
- Perform signal processing and decision-making on the Raspberry Pi board
  - Prevents signal delays that could occur when using cloud services
- Communicate directly with other vehicles via RF antenna
