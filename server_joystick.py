import socket
import struct
import time
import cv2
import numpy
import pymavlink.mavutil as mavutil
from multiprocessing import Value
import multiprocessing
from multiprocessing import Value
import argparse

class Carame_Accept_Object:
    def __init__(self,S_addr_port=("",65432)):
        self.resolution=(640,480)       
        self.img_fps=30
        self.addr_port=S_addr_port
        self.Set_Socket(self.addr_port)
 
    def Set_Socket(self,S_addr_port):
        self.server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
        self.server.bind(S_addr_port)
        self.server.listen(5)

def getValue(cam_triggered):
    trigger_threhold = 1500
    while True:
        the_connection = mavutil.mavlink_connection('/dev/ttyUSB0',baud=921600)
        the_connection.wait_heartbeat()
        cam_triggered_value = the_connection.messages['RC_CHANNELS'].chan9_raw
        if cam_triggered_value > trigger_threhold:
            with cam_triggered.get_lock():
                cam_triggered.value = 1
        else:
            with cam_triggered.get_lock():
                cam_triggered.value = 0


def check_option(object,client):
    info=struct.unpack('lhh',client.recv(12))
    if info[0]>888:
        object.img_fps=int(info[0])-888         
        object.resolution=list(object.resolution)
       
        object.resolution[0]=info[1]
        object.resolution[1]=info[2]
        object.resolution = tuple(object.resolution)
        return 1
    else:
        return 0

def RT_Image(object,client,D_addr,filename,cam_triggered):
    if(check_option(object,client)==0):
        return
    camera=cv2.VideoCapture(0)                               
    img_param=[int(cv2.IMWRITE_JPEG_QUALITY),object.img_fps] 
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = 30
    camera.set(cv2.CAP_PROP_FPS, fps)
    
    print(width, height, fps)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    size = (int(width),int(height))
    out=cv2.VideoWriter(filename, fourcc, int(fps), size)
    while(1):
        time.sleep(0.1)             
        _,object.img=camera.read()  

        cam_triggered_value = cam_triggered.value
        if int(cam_triggered_value) > 0:

            out.write(object.img)

        object.img=cv2.resize(object.img,object.resolution)    
        _,img_encode=cv2.imencode('.jpg',object.img,img_param)  
        img_code=numpy.array(img_encode)                        
        object.img_data=img_code.tostring()                     
        try:
            client.send(struct.pack("lhh",len(object.img_data),object.resolution[0],object.resolution[1])+object.img_data)
        except:
            camera.release()     
            out.release()
            return
 
if __name__ == '__main__':

    arg = argparse.ArgumentParser(description='load video and save')
    arg.add_argument('--filename',type=str,default='out.avi',help='filename and filepath')
    opt = arg.parse_args()

    camera=Carame_Accept_Object()
    cam_triggered = Value('d', 0)
    while (1):
        client,D_addr=camera.server.accept()
        clientProcess=multiprocessing.Process(None,target=RT_Image,args=(camera,client,D_addr,opt.filename,cam_triggered,))
        valueProcess = multiprocessing.Process(None,target=getValue, args=(cam_triggered, ))
        clientProcess.start()
        valueProcess.start()