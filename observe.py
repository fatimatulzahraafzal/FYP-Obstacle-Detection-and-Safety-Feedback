# Import the following libraries
import argparse
import time
import pickle
import socket
import struct
import cv2
import pyrealsense2 as rs
import numpy as np
import threading
import pyglet
pyglet.options["headless"] = True
import pyglet.media
import pyttsx3

input_list = []

# Define the server address :Jetson Orin IP and port
server_ip = '172.20.10.6'  
server_port = 50000  

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Python Text to Speech Engine for generating speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate",150)
    engine.say(text)
    engine.runAndWait()

# Audio queue is maintained and updated after each obstacle
class AudioQueue(threading.Thread):
    def __init__(self, obstacle):
        threading.Thread.__init__(self)
        self.counter = 0
        self.dist = 10

    # Update method
    def update(self, dist, x, y, name=''):
        self.dist = dist
        self.x = x
        self.y = y
        self.name = name

    # Run method
    def run(self):
        while True:
            self.counter = (self.counter + 1) % 5
            if self.dist < 5:
                # In observe mode, speech is generated when the obstacle class is given
                if self.name != '' and self.counter == 0:
                    self.generate_sound() #generate a brief earcon
                    self.generate_speech() #generate speech
            # In this case, no obstacle detected
            else:
                time.sleep(0.5)


    def generate_speech(self):
        
        # Create a new pyglet Player object
        player = pyglet.media.Player()
        # Start playing the audio 
        player.play()

        # Calculate normalized audio coordinates and distance
        audio_x = (self.x - 320) / 213  # -1.5 to 1.5
        audio_y = (self.y - 240) / 240
        audio_z = self.dist / 3

        # Set the position of the player in 3D space based on audio coordinates and distance
        player.position = (audio_x, audio_y, audio_z)
        # Calculate the pitch of the speech based on the distance
        pitch = max(0.8, 1 / self.dist)
        # Set the pitch of the player
        player.pitch = pitch

        time.sleep(1)
        # Print debug information about the player's position
        print("position:", player.position)  # Debug purposes
        print("position:", player.position[0])  # Debug purposes
        # Print debug information about audio_x
        print(audio_x)  # Debug purposes
        # Print debug information about self.name
        print("name: ", self.name)  # Debug purposes
        # Print debug information about self.dist
        print("dist: ", round(self.dist, 2))  # Debug purposes

        # Check if self.name is empty
        if self.name == "":
            print("empty")  # Debug purposes
            # Speak the text "obstacle"
            speak_text("obstacle")
        else:
            print("not empty")  # Debug purposes
            # Speak the value of self.name
            speak_text(self.name)

        # Check if audio_x is less than 0
        if audio_x < 0:
            print("obstacle to the left,")  # Debug purposes
            # Speak the text "to the left"
            speak_text("to the left")
            # Speak the rounded value of self.dist
            speak_text(round(self.dist, 2))
            # Speak the text "metre"
            speak_text("metre")
        else:
            print("obstacle to the right")  # Debug purposes
            # Speak the text "to the right"
            speak_text("to the right,")
            # Speak the rounded value of self.dist
            speak_text(round(self.dist, 2))
            # Speak the text "metre"
            speak_text("metre")


    def generate_sound(self):
        # Create a new pyglet Player object
        player = pyglet.media.Player()
        # Start playing the audio
        player.play()

        # Calculate normalized audio coordinates and distance
        audio_x = (self.x - 320) / 213  # -1.5 to 1.5
        audio_y = (self.y - 240) / 240
        audio_z = self.dist / 4

        # Calculate audio duration based on distance, limiting to a maximum of 0.10 seconds
        audio_dur = min(self.dist / 5, 0.10)
        # Calculate idle duration based on distance
        idle_dur = self.dist / 2

        # Calculate frequency based on audio distance
        freq = 700 - audio_z * 250

        # Generate a Triangle waveform with specified duration and frequency
        wave = pyglet.media.synthesis.Triangle(audio_dur, freq)
        # Set the position of the player in 3D space based on audio coordinates and distance
        player.position = (audio_x, audio_y, audio_z)

        # Queue the generated waveform for playback
        player.queue(wave)

        # Sleep for the total duration of audio playback and idle time
        time.sleep(audio_dur + idle_dur)


def transmit_frames():
    audio_queue = AudioQueue(None)
    audio_queue.start()

    # Configure RealSense camera
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print('Enabling RealSense camera')
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensor_dep = profile.get_device().first_depth_sensor()
    sensor_dep.set_option(rs.option.min_distance, 100)
    sensor_dep.set_option(rs.option.laser_power, 100)
    sensor_dep.set_option(rs.option.receiver_gain, 18)
    sensor_dep.set_option(rs.option.confidence_threshold, 1)
    sensor_dep.set_option(rs.option.noise_filtering, 2)

    depth_scale = sensor_dep.get_depth_scale()
    print(depth_scale) # Debug purposes

    align_to = rs.stream.color
    align = rs.align(align_to)

    print('Enabled RealSense camera')

    try:
        global frame
        frame = 1
        
        print('Trying to send data...') # Debug purposes
        client_socket.connect((server_ip, server_port)) # Connect to the receiver

        depth_scale_data = pickle.dumps(depth_scale)  # Convert the depth scale to pickle

        depth_scale_data_len = struct.pack("Q", len(depth_scale_data))
        client_socket.sendall(depth_scale_data_len) # Send the length of the depth_scale data

        client_socket.sendall(depth_scale_data)  # Send the depth_scale data
        print('Depth scale sent')

        while True:
            t0 = time.perf_counter() # Time stanp

            frames = pipeline.wait_for_frames() # Wait for frames
            aligned_frames = align.process(frames) # Align the frames to get synchronized
            color_frame = aligned_frames.get_color_frame() # Retrieve the color frame from the aligned frames
            depth_frame = aligned_frames.get_depth_frame() # Retrieve the depth frame from the aligned frames

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data()) # Convert the color frame to np array
            depth_image = np.asanyarray(depth_frame.get_data()) # Convert the depth frame to np array

            color_data = pickle.dumps(color_image) # Convert the color images
            depth_data = pickle.dumps(depth_image) # Convert the depth images

            color_data_len = struct.pack("L", len(color_data)) # Length of the color data
            client_socket.sendall(color_data_len) # Send the length of the color data
           
            client_socket.sendall(color_data)  # Send the color data

            depth_data_len = struct.pack("L", len(depth_data)) # Length of the depth data
            client_socket.sendall(depth_data_len) # Send the length of the depth data
            
            client_socket.sendall(depth_data) # Send the depth data
            #print('Coherent Frames sent') # Debug purposes
            
            t1 = time.perf_counter() # Time stamp
            input_list.append(1E3 * (t1 - t0)) # Debug purposes to calculate time lag
            
            #print("average--> input:", sum(input_list)/len(input_list)) # Debug purposes

            # Receive a response from Orin
            response = client_socket.recv(1024).decode()
            #print("Received response from Jetson Orin:", response) # Debug purposes

            obs = response.split(',') # Split the response
            dist, x, y, name = float(obs[0]), float(obs[1]), float(obs[2]), obs[3] # Extract the individual values

            audio_queue.update(dist, x, y, name) # Update the audio queue

    except Exception as e:
        print('Error occurred:', e)
    finally:
        pipeline.stop()
        client_socket.close()


if __name__ == '__main__':
    transmit_frames()