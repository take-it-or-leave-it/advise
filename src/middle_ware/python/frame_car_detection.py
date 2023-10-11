
import cv2 as cv
import numpy as np
import sys
import time

"""

last update : 23-10-10 19:52 / ByeongChan
https://github.com/take-it-or-leave-it/test_camera/blob/main/test_multi_camera/multi_cam_concat.py
이거 참고되었고, 나중에 함수화 하겠음

"""
def Cam_index_test():
    #### input camera index #####
    index = 4
    
    #############################

    cap1 = cv.VideoCapture(index)
    width = cap1.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('output'+str(index)+'.avi', fourcc, 15.0, (int(width), int(height)))
    cnt = 0
    while(cap1.isOpened()):
        ret1, frame1 = cap1.read()    # Read 결과와 frame
        cnt+=1
        if(ret1 == False) : 
            break;   
        print(cnt)
        #cv.imshow('frame_color',frame1)
        out.write(frame1)
        if(cnt == 100):
            break
        if cv.waitKey(1) == ord('q'):
            
            break
     
    cap1.release()    
    out.release()   

def Multi_Cam_test():



    #### input cameras index #####
    index1 = 0
    index2 = 2
    index3 = 4
    #############################
    
    cap1 = cv.VideoCapture(index1)
    cap1.set(cv.CAP_PROP_FPS, 15.0)
    cap1.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G'))

    cap2 = cv.VideoCapture(index2)
    cap2.set(cv.CAP_PROP_FPS, 15.0)
    cap2.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G'))

    cap3 = cv.VideoCapture(index3)
    cap3.set(cv.CAP_PROP_FPS, 15.0)
    cap3.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc('M','J','P','G'))

    width = 640
    height = 480
    print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('multi output.avi', fourcc, 15.0, (1280, 720))
    cnt = 0
    while(1):
        ##for measure time gap between cam 1 and cam 2
        start = time.time()
        
        ret1, frame1 = cap1.read()    # Read 결과와 frame
        ret2, frame2 = cap2.read()      
        ret3, frame3 = cap3.read()
        

        end = time.time()
        print(f"{end - start:.5f} sec")

        cnt+=1
            
        #time.sleep(0.1)
        print(cnt)
        #cv.imshow('frame_color',frame1)
        res_frame = cv.hconcat([frame1,frame2,frame3])

        start = time.time()
        res_frame = cv.resize(res_frame, (1280, 720),  interpolation=cv.INTER_LINEAR) # 스케일 팩터 이용
        print(res_frame.shape)
        end = time.time()
        print(f"resize time {end - start:.5f} sec")
        out.write(res_frame)
        if(cnt == 50):
            break
        if cv.waitKey(1) == ord('q'):
            
            break
     
    cap1.release() 
    cap2.release()    
    cap3.release()    
    out.release()   

#Multi_Cam_test()
#Cam_index_test()