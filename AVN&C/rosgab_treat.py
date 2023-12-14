import rosbag
from scipy.spatial.transform import Rotation as R
import numpy as np

with rosbag.Bag('output.bag', 'w') as outbag:
    i = 0
    for topic, msg, t in rosbag.Bag('2023-12-13-13-33-10.bag').read_messages():

        # This also replaces tf timestamps under the assumption 
        # that all transforms in the message share the same timestamp
        if topic == "/vrpn_client_node/QCar/pose":
            
            msg.pose.position.x = msg.pose.position.x*0.55
            msg.pose.position.y = msg.pose.position.y*0.211
            a = msg.pose.orientation.x
            b = msg.pose.orientation.y
            c = msg.pose.orientation.z
            d = msg.pose.orientation.w
            r = R.from_quat([a, b, c, d])
            rot = r.as_euler('zyx', degrees=True)
            
            msg.pose.position.z = rot[0]
            print('vicon',rot[0])
            outbag.write(topic, msg, t)
        if topic == "/qcar/disired_pose" and msg.vector:
            #print('qcar',msg.vector.x,msg.vector.y)
            msg.vector.x = -msg.vector.x + 0.658
            msg.vector.y =  msg.vector.y + 0.086
            msg.vector.z = -msg.vector.z*180/np.pi +6-0.28
            print('qcar',msg.vector.z)
            outbag.write(topic, msg, t)
        else:
            outbag.write(topic, msg,t)

#qcar -0.7012369147773463 -0.25274203686536584
#vicon -0.0959025586563887 -0.7930707286205784 
#qcar -0.15300529959634654 -0.03919202140579321
#vicon 0.8344726809334242 0.006279246265176086