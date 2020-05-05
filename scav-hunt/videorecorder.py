import cv2
from threading import Thread
import os
import time

RESOLUTION = (640,480)
FRAME_RATE = 30

class VideoRecorder:
    def __init__(self,path,runid,show=False):
        print('Init Video Recorder')
        self.runid = runid
        self.resolution = RESOLUTION
        self.framerate = FRAME_RATE
        self.stream_init()
        self.output_path = path
        self.show = show
        

    def stream_init(self):
        """Function to capture video device and initialize input stream"""
        print('Initializing Video stream')
        self.stream = cv2.VideoCapture(0)
        time.sleep(0.01) #camera warmup
        #Read the first frame
        if not self.stream:
            print('Camera not ready')
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False        
        

    def output_init(self):
        """Function to initialize output stream"""
        self.output_file_name = self.generate_new_file_name(self.runid)
        self.output = cv2.VideoWriter(self.output_file_name,cv2.VideoWriter_fourcc(*'MJPG'),self.framerate,self.resolution)
        print(self.output_file_name)
        

    def generate_new_runid(self,runid):
        """Function to generate runid used in creating the output file name"""
        self.runid += 1
        return self.runid

    def generate_new_file_name(self,runid):
        """Function to generate new output file name"""
        new_runid = self.generate_new_runid(runid)
        return 'output'+'_'+str(new_runid)+ '.avi'   
    
    def start_recording(self):
        """Function to start recording on a new thread"""
        self.output_init() #initialize output stream
        Thread(target=self.record,args=()).start()
        return self
        
    def record(self):
        """Function to record input stream to disk"""
        print('Started recording')
        if not self.stream.isOpened():
            self.stream_init()
        while True:
            if self.stopped:
                return
            else:
                while(self.stream.isOpened()):
                    ret, frame = self.stream.read()
                    if ret==True:
                        # write the frame
                        try:
                            self.output.write(frame)
                        except:
                            print('Caught stream write fault')
                            pass
                        if self.show:
                            cv2.imshow('frame',frame)
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break
                    else:
                        break


    def stop_recording(self):
        """Function to stop recording"""
        self.stopped = True
        self.stream.release() 
        time.sleep(0.1) #sleeping to prevent segmentation fault while the thread is still writing before releasing the output stream
        self.output.release()
        if self.show:
            cv2.destroyAllWindows()
        print('Stopped recording')
        self.move_file_to_path()


    def move_file_to_path(self):
        """Function to move the video recorded in scav-hunt folder to the videos/cone_color folder"""
        filepath = os.path.join(os.getcwd(),self.output_file_name)
        destination_path = os.path.join(self.output_path,self.output_file_name)
        os.rename(filepath, destination_path)
        print('{} file saved at {}'.format(self.output_file_name, self.output_path))