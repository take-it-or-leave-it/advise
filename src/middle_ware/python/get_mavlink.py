import rospy
from pymavlink import mavutil
from advise.msg import MavlinkMsg, NearCarMsg

#for GPS and ground speed 
gps_key = "GPS_RAW_INT"

#for RADAR
radar_key = ""

mavlink_publisher = rospy.Publisher("advise/mavlink",MavlinkMsg)
nearcar_publisher = rospy.Publisher("advise/near_car",NearCarMsg)

if __name__ == "__main__":
    
    # launch에서 파라미터 받기(pixhawk 연결 포트), default는 COM4
    px4_port = rospy.get_param('px4_port','COM4')

    rospy.init_node("get_mavlink_node")
    rate = rospy.Rate(5.0)

    connection = mavutil.mavlink_connection(px4_port)

	while not rospy.is_shutdown():
		gps_value = connection.recv_match(
            blocking=True,
            type=gps_key
        )
        radar_value = connection.recv_match(
            blocking=True,
            type=radar_key
        )
        
        """
            TODO : gps값으로 서버에 쿼리, 주변차량 정보 받기(post 및 mqtt)

            이 정보로 near car table topic 만들기

        """

        """
            TODO : gps, ground speed, radar 값을 모아서 topic으로 publish 
        """

        rate.sleep()
        
	