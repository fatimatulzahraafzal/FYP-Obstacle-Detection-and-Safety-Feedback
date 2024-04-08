import socket
import struct
import pickle
import cv2
import argparse
import time
from pathlib import Path
import ctypes
import otsu
from numpy import random
import pyrealsense2 as rs
import numpy as np

#time stamps are stored in these
ENABLE_YOLO_DETECT = True
input_list = []
filter_list = []
contour_list = []
yolo_list = []
output_list = []
total_list = []


# Define the server address (Orin IP and port)
server_ip = ''  # Listen on all available interfaces
server_port = 50000  # Choose a desired port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# X11 multithread support
ctypes.CDLL('libX11.so.6').XInitThreads()

if ENABLE_YOLO_DETECT:
    from ncnn.utils import draw_detection_objects
    from yolov8 import YoloV8s


# Obstacle class
class Obstacle():
    def __init__(self, dist, xyxy, name, prob, mid_xy):
        self.dist = dist  # Distance of the obstacle
        self.xyxy = xyxy  # Bounding box coordinates of the obstacle
        self.name = name  # Name of the obstacle
        self.prob = prob  # Probability associated with the obstacle
        self.mid_xy = mid_xy  # Midpoint coordinates of the obstacle

    def __lt__(self, obj):
        return self.dist < obj.dist  # Less than comparison based on distance

    def __gt__(self, obj):
        return self.dist > obj.dist  # Greater than comparison based on distance

    def __le__(self, obj):
        return self.dist <= obj.dist  # Less than or equal to comparison based on distance

    def __ge__(self, obj):
        return self.dist >= obj.dist  # Greater than or equal to comparison based on distance

    def __eq__(self, obj):
        return self.dist == obj.dist  # Equality comparison based on distance

    def __repr__(self):
        return '{' + str(self.dist) + ', ' + self.dist + ', ' + self.prob + '}'  # String representation of the obstacle object

# Helper function for distance sampling
def sample_distance(depth_image, mid_x, mid_y):
    global depth_scale
    window = 2
    # Extract a sample window from the depth image
    sample_depth = depth_image[mid_y-window:mid_y+window, mid_x-window:mid_x+window].astype(float)
    # Calculate the mean depth within the window
    dist, _, _, _ = cv2.mean(sample_depth)
    # Apply the depth scale factor to convert to real-world distance
    dist = dist * depth_scale
    return dist

# Helper function for drawing labels on an image
def draw_label(image, text, x0, y0, x1, y1, color=(0,0,0)):
    # Draw a rectangle around the object
    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0))
    # Calculate the size of the label text
    label_size, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Calculate the position of the label text
    x = x0
    y = y0 - label_size[1] - baseLine
    # Adjust the position if it goes beyond image boundaries
    if y < 0:
        y = 0
    if x + label_size[0] > image.shape[1]:
        x = image.shape[1] - label_size[0]
    # Draw a filled rectangle as the background for the label
    cv2.rectangle(image, (int(x), int(y)), (int(x + label_size[0]), int(y + label_size[1] + baseLine)), color, -1)
    # Draw the label text on top of the background
    cv2.putText(image, text, (int(x), int(y + label_size[1])), cv2.LINE_AA, 0.5, (255, 255, 255))

# Helper function for drawing detection objects on images
def draw_detection_objects(image, depth_colormap, objects):
    for idx, obj in enumerate(objects):
        # Extract the coordinates of the object's bounding box
        x0, y0, x1, y1 = obj.xyxy[0], obj.xyxy[1], obj.xyxy[2], obj.xyxy[3]
        # Create the label text with the object's name and distance
        text = f'{obj.name} {obj.dist:.2f}m'
        # Determine the color for drawing the label based on the object's index and probability
        if idx < 2:
            color = (50,50,150)  # Blue color for top two objects
        else:
            color = (0,150,100) if obj.prob < 0 else (0,0,0)  # Green color if probability is negative, otherwise black
        # Draw the label on the color image and depth colormap
        draw_label(image, text, x0, y0, x1, y1, color)
        draw_label(depth_colormap, text, x0, y0, x1, y1, color)


# Define the server address (Orin IP and port)
server_ip = ''  # Listen on all available interfaces
server_port = 50000  # Choose a desired port number


def receive_frames():

    # Initialize YOLO
    if ENABLE_YOLO_DETECT:
        print('Loading model')
        net = YoloV8s(target_size=32*6, prob_threshold=0.25, nms_threshold=0.45, num_threads=2, use_gpu=True, )

    global device, model, half, depth_scale
    
    # Bind the socket to the server address
    server_socket.bind((server_ip, server_port))
    # Listen for incoming connections
    server_socket.listen(1)
    
    print('Waiting for connection...')
    
    # Accept a client connection from Nano
    client_socket, client_address = server_socket.accept()
    print("Connected with Jetson Nano:", client_address)
    
    # Receive the length of the depth_scale data
    depth_scale_data_len = client_socket.recv(8)
    depth_scale_data_len = struct.unpack("Q", depth_scale_data_len)[0]

    # Receive the depth_scale data
    depth_scale_data = b''
    while len(depth_scale_data) < depth_scale_data_len:
        data = client_socket.recv(4096)
        depth_scale_data += data

    # Unpickle the depth_scale data
    depth_scale = pickle.loads(depth_scale_data)
    print('Received depth scale:', depth_scale)


    try:
        global frame
        frame = 1
        
        while True:
            t0 = time.perf_counter()
            # Receive the length of the color data
            color_data_len = client_socket.recv(struct.calcsize("L"))
            color_data_len = struct.unpack("L", color_data_len)[0]

            # Receive the color data
            color_data = b""
            while len(color_data) < color_data_len:
                packet = client_socket.recv(color_data_len - len(color_data))
                if not packet:
                    break
                color_data += packet

            # Deserialize the color data
            img = pickle.loads(color_data)

            # Receive the length of the depth data
            depth_data_len = client_socket.recv(struct.calcsize("L"))
            depth_data_len = struct.unpack("L", depth_data_len)[0]

            # Receive the depth data
            depth_data = b""
            while len(depth_data) < depth_data_len:
                packet = client_socket.recv(depth_data_len - len(depth_data))
                if not packet:
                    break
                depth_data += packet

            # Deserialize the depth data
            depth_img = pickle.loads(depth_data)
            
            t1 = time.perf_counter()  #after receiving the data

            im0 = img.copy()
            invalid = np.full((480,640),65536, dtype=np.uint16)
            #depth_img = np.where(depth_img[:,:] == [0,0], invalid, depth_img)
            depth_img = cv2.medianBlur(depth_img,5)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            
            t2 = time.perf_counter()  
            
            obstacles, yolo_obstacles = [], []
            contours = []
            otsu_img = (depth_img // 100).astype(np.uint8).clip(0,255)
            hist = cv2.calcHist([otsu_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            
            try:
                # Attempt to execute the following code block
                raise Exception  # Raise an exception (for demonstration purposes)
                thresholds = sorted(otsu.modified_TSMO(hist, M=64, L=256))  # Calculate sorted thresholds using OTSU algorithm
                if len(thresholds) < 1:
                    raise Exception  # Raise an exception if there are no thresholds

                thresh_a, layer = 0, 1  # Initialize threshold variables
                for thresh_b in thresholds:
                    depth_range = cv2.inRange(otsu_img, thresh_a, thresh_b)  # Create a binary image within the threshold range
                    h, w = depth_range.shape[0:2]  # Get the height and width of the depth range image
                    cv2.rectangle(depth_range, (0, 0), (w, h), (0), 50)  # Draw a thick white rectangle on the depth range image
                    contours_range, _ = cv2.findContours(depth_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the depth range image
                    for c in contours_range:
                        contours.append(c)  # Append the found contours to the overall contours list
                    if layer > 2:
                        break  # Break the loop if layer exceeds 2
                    thresh_a = thresh_b  # Update the starting threshold for the next layer
                    layer = layer + 1  # Increment the layer counter
            except:
                # Execute the following code block if an exception occurs - Linear Thresholding for Backup
                c_start, c_step, c_levels, layer = 0.0, 0.5, 5, 1  # Initialize threshold variables
                for i in range(c_levels):
                    depth_range = cv2.inRange(depth_img, c_start / depth_scale, (c_start + c_step) / depth_scale)  # Create a binary image within the threshold range
                    h, w = depth_range.shape[0:2]  # Get the height and width of the depth range image
                    cv2.rectangle(depth_range, (0, 0), (w, h), (0), 20)  # Draw a rectangle on the depth range image
                    contours_range, _ = cv2.findContours(depth_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the depth range image
                    for c in contours_range:
                        contours.append(c)  # Append the found contours to the overall contours list
                  
            for c in contours:
                # Calculate size and bounding rectangle of the contour
                size = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)

                # Skip if the bounding rectangle is too small or too large
                if w < 50 or h < 50 or w >= 640 or h >= 480:
                    continue

                # Calculate the centroid of the contour
                M = cv2.moments(c)
                mid_x, mid_y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

                # Sample the distance at the centroid position
                dist = sample_distance(depth_img, mid_x, mid_y)

                if dist > 0:
                    # Create an obstacle object with the calculated properties
                    obs = Obstacle(dist, [x, y, x + w, y + h], '', -1, [mid_x, mid_y])
                    obstacles.append(obs)  # Append the obstacle to the list of obstacles

            t3 = time.perf_counter()

            # YOLO inference process
            if ENABLE_YOLO_DETECT:
                objects = net(img)

                for obj in objects:
                    # Skip if the object's probability is below a threshold
                    if obj.prob < 0.1:
                        continue

                    xyxy = [obj.rect.x, obj.rect.y, obj.rect.x + obj.rect.w, obj.rect.y + obj.rect.h]

                    # Calculate the centroid of the object
                    mid_x, mid_y = round(int(xyxy[0] + xyxy[2]) / 2), round(int(xyxy[1] + xyxy[3]) / 2)

                    # Sample the distance at the centroid position
                    dist = sample_distance(depth_img, mid_x, mid_y)
                    name = net.class_names[int(obj.label)]

                    if dist > 0:
                        # Create an obstacle object with the calculated properties
                        obs = Obstacle(dist, xyxy, name, obj.prob, [mid_x, mid_y])
                        obstacles.append(obs)  # Append the obstacle to the list of obstacles
                        yolo_obstacles.append(obs)  # Append the obstacle to the YOLO-specific list of obstacles

            obstacles.sort()  # Sort the obstacles based on their distance
            yolo_obstacles.sort()  # Sort the YOLO-specific obstacles based on their distance
            draw_detection_objects(im0, depth_colormap, obstacles)  # Draw the detected objects on the image

            
            t4 = time.perf_counter() #time stamp5 (the end)
            
            if len(obstacles) > 0:
                msg = str(obstacles[0].dist) + ',' + str(obstacles[0].mid_xy[0]) + ',' + str(obstacles[0].mid_xy[1]) + ',' + obstacles[0].name
                # obs[0] does not have name but yolo_obs[0] has name and dist are close
                if len(yolo_obstacles) > 0 and obstacles[0].name == '' and yolo_obstacles[0].name != '' and yolo_obstacles[0].dist - obstacles[0].dist < 0.1:
                    msg = str(yolo_obstacles[0].dist) + ',' + str(yolo_obstacles[0].mid_xy[0]) + ',' + str(yolo_obstacles[0].mid_xy[1]) + ',' + yolo_obstacles[0].name
                #print(msg)
            
            # Send the message response to Nano
            client_socket.sendall(msg.encode())
            
            t5 = time.perf_counter() #time stamp6 (sending output)

            input_list.append(1E3 * (t1 - t0))
            filter_list.append(1E3 * (t2 - t1))
            contour_list.append(1E3 * (t3 - t2))
            yolo_list.append(1E3 * (t4 - t3))
            output_list.append(1E3 * (t5 - t4))
            total_list.append(1E3 * (t5 - t0))   
            
            # Debug Purposes 
            print("input:", 1E3 * (t1 - t0) , 
                  "filter", 1E3 * (t2 - t1), 
                  "contour", 1E3 * (t3 - t2) , 
                  "yolo" , 1E3 * (t4-t3) , 
                  "output" , 1E3 * (t5-t4) , 
                  "total" , 1E3 * (t5-t0))     
                  
            print("average--> input:", sum(input_list)/len(input_list) , 
                  "filter", sum(filter_list)/len(filter_list), 
                  "contour", sum(contour_list)/len(contour_list) , 
                  "yolo" , sum(yolo_list)/len(yolo_list) , 
                  "output" , sum(output_list)/len(output_list) , 
                  "total" , sum(total_list)/len(total_list))
            #frame = frame + 1 # Debug Purposes 
            #print(frame) # Debug Purposes 
 
            cv2.imshow("YOLOv8 result", im0) # Debug Purposes 
            cv2.imshow("Depth result", cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)) # Debug Purposes 
            cv2.waitKey(1)
            

    except Exception as e:
        print('Error occurred:', e)
    finally:
        client_socket.close()
        server_socket.close()

if _name_ == '_main_':
    receive_frames()