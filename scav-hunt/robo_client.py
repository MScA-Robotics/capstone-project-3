import os
import sys, errno
import socket,  time, datetime
import ssl
import sys
import struct
import logging


import robotics_pb2


def connection(host, port, login, password, split_id, filename):
    # logger for client-server connection
    if not os.path.exists('./log_client/'):
        os.makedirs('./log_client/')
    log_file_name = './log_client/' + datetime.datetime.now().strftime('Treasure_Hunt_Challenge' + '_%Y_%m_%d_%H_%M.log')     
    logger = logging.getLogger('server')
    hdlr = logging.FileHandler(log_file_name)
    formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.DEBUG)
    logger.info( u'Starting...' )
    
    # Create a socket (SOCK_STREAM means a TCP socket)        
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)    
    # WRAP SOCKET
    wrappedSocket = ssl.wrap_socket(sock, ssl_version=ssl.PROTOCOL_TLSv1_1)

    try:        
        # Connect to server and send data
        wrappedSocket.connect((host, port))  
        
        # fill proto fields      
        r_data = robotics_pb2.Robotics_Data()
        r_data.login = login
        r_data.enc_password = password        
        r_data.split_id = split_id

        # read data from log file and fill proto field for log data
        file_open = open(filename)
        s_data = file_open.readlines()        
        for s in s_data:
            r_data.log_data.append(s) 
        
        # send to server login, password, split_id and log file data
        data = r_data.SerializeToString()
        wrappedSocket.sendall(struct.pack(">H", len(data)))
        wrappedSocket.sendall(data)

        # check if authentification is successfull
        rec = wrappedSocket.recv(2)
        reply_len = struct.unpack(">H", rec)[0] 
        rec = wrappedSocket.recv(reply_len)
        signal = robotics_pb2.LoginReply()
        signal.ParseFromString(rec) 
        if signal.connection_status == robotics_pb2.LoginReply.OK:
            logger.info(u'Successfull authentication') 
        else: 
            if signal.connection_status == robotics_pb2.LoginReply.BAD_LOGIN_OR_PASSWORD:
                logger.info(u'Authentication request FAILED. Bad login or password') 
                return False
            else:
                if signal.connection_status == robotics_pb2.LoginReply.BAD_SPLIT_ID:
                    logger.info(u'Authentication request FAILED. Bad split id') 
                    return False

        # wait for message from server if data receiving is successfull, 20 sec. maximum
        # 
        finish_time = time.time() + 20.0
        while(True):
            rec = wrappedSocket.recv(2)
            reply_len = struct.unpack(">H", rec)[0] 
            rec = wrappedSocket.recv(reply_len)
            signal = robotics_pb2.Signal()
            signal.ParseFromString(rec) 
            if 1 == signal.end:
                logger.info( u'Your data has been streamed successfully') 
                return True
            else:
                if 0 == signal.end:
                    logger.info( u'Error: possibly your file has bad format')
                    return False                    
                else:
                    if time.time() > finish_time:
                        logger.info( u'Waiting time is over. Something goes wrong') 
                        return False
        
    except RuntimeError:
            logger.exception('')
    except Exception:
        logger.exception('')    
    finally:    
        wrappedSocket.close()




if __name__ == '__main__':
    HOST, PORT = 'datastream.ilykei.com', 30078
    login = 'Your Login'
    password = 'Your Password for Data Stream Assignments'
    split_id = 19
    filename = '../gpg/log/20200304_165721_s44.log'
    connection(HOST, PORT, login, password, split_id, filename)
