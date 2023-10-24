import rospy
from advise.msg import PlayAudioMsg
import playsound

def play_audio(audio_msg) :

    if(audio_msg.dilema_id==1) :
        if(audio_msg.is_receiver) :
            if(audio_msg.is_safe):
                playsound.playsound('./asset/그대로진행.mp3')
            else: 
                playsound.playsound('./asset/양보.mp3')
        else :
            if(audio_msg.is_safe):
                playsound.playsound('./asset/진입시안전.mp3')
            else: 
                playsound.playsound('./asset/진입시위험.mp3')
    elif(audio_msg.dilema_id==2) :
        if(audio_msg.is_receiver) :
            if(audio_msg.is_safe):
                playsound.playsound('./asset/그대로진행.mp3')
            else: 
                playsound.playsound('./asset/주의.mp3')
        else :
            if(audio_msg.is_safe):
                playsound.playsound('./asset/그대로진행.mp3')
            else: 
                playsound.playsound('./asset/주의.mp3')
    elif(audio_msg.dilema_id==3) :
        if(audio_msg.is_receiver) :
            if(audio_msg.is_safe):
                playsound.playsound('./asset/그대로진행.mp3')
            else: 
                playsound.playsound('./asset/주의.mp3')
        else :
            if(audio_msg.is_safe):
                playsound.playsound('./asset/앞차량경고.mp3')
            else: 
                playsound.playsound('./asset/주의.mp3')

if __name__ == "__main__":
    
    rospy.init_node("play_audio_node")

    data_receiver = rospy.Subscriber("advise/request/play_audio",PlayAudioMsg,play_audio)

    rate = rospy.Rate(200.0)

	while not rospy.is_shutdown():
        
        rate.sleep()
        
	