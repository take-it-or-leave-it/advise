import rospy
import cv2
from advise.msg import FrameCarMsg

framecar_publisher = rospy.Publisher("advise/frame_car",FrameCarMsg)

if __name__ == "__main__":
    rospy.init_node("frame_car_node")
    rate = rospy.Rate(5.0)

    toSendMsg = FrameCarMsg()

    while not rospy.is_shutdown():
        toSendMsg.timeStamp = rospy.Time.now()

        """
            카메라 처리하기







            
        """
        
        # 앞차량 존재 여부는 아래 값으로 저장
        toSendMsg.frontCar = True
        # 뒷차량 존재 여부는 아래 값으로 저장
        toSendMsg.rearCar = True
        # 뒷차량과의 거리는 아래 값으로 저장
        toSendMsg.rearDistance = 0.01
        # 토픽으로 보내기 (30msTasks에서 받을 예정임)
        framecar_publisher.publish(toSendMsg)
    
        rate.sleep()