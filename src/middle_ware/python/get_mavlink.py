import rospy
from pymavlink import mavutil
from advise.msg import MavlinkMsg, InfoTable
import pika

#for GPS and ground speed 
gps_key = "GPS_RAW_INT"

#for RADAR
radar_key = "DISTANCE_SENSOR"

HOST_NAME = "192.168.238.233"
user_credentials = pika.PlainCredentials('rasp', '1234')
QUEUE_NAME = "advise.queue"

CAR_ID=1

connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=HOST_NAME,
            credentials=user_credentials
            )

        )
channel = connection.channel()
channel.queue_declare(queue=QUEUE_NAME)

def send_mqtt(toSend):
    channel.basic_publish(exchange='advise.exchange', routing_key="advise.key", body=json.dumps(toSend))
     
mavlink_publisher = rospy.Publisher("advise/mavlink",MavlinkMsg)
nearCar_publisher = rospy.Publisher("advise/near_car",InfoTable)
mqttCar_subscriber = rospy.Subscriber("advise/request/push_mqtt",InfoTable,)


def receive_mqtt(ch, method, properties, body):
    nearCar_publisher.publish(body)

mavlink_to_send = MavlinkMsg()
near_to_send = NearCarMsg()


if __name__ == "__main__":
    
    # launch에서 파라미터 받기(pixhawk 연결 포트), default는 '/dev/ACM0'
    px4_port = rospy.get_param('px4_port','/dev/ACM0')

    rospy.init_node("get_mavlink_node")
    rate = rospy.Rate(200.0)

    connection = mavutil.mavlink_connection(px4_port)
    channel.basic_consume(receive_mqtt,
                      queue=f'advise{CAR_ID}.queue',
                      no_ack=True)

	while not rospy.is_shutdown():

        now = rospy.get_rostime()

        radar_value = connection.recv_match(
            blocking=False,
            type=radar_key
        )
        gps_value = connection.recv_match(
                blocking=False,
                type=gps_key
        )
        mavlink_to_send.timeStamp = now
        mavlink_to_send.lat = gps_value.lat
        mavlink_to_send.lon = gps_value.lon
        mavlink_to_send.vel = gps_value.vel
        mavlink_to_send.frontDistance = radar_value.current_distance
        mavlink_publisher.publish(mavlink_to_send)

        rate.sleep()

    connection.close()
        
	