import socket
import paho.mqtt.client as mqtt
import base64
import struct
import json
import threading
import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def check_image_similarity(img1, img2):
    # Resize both for comparison
    
    img1_resized = cv2.resize(img1, (256, 256))
    img2_resized = cv2.resize(img2, (256, 256))
    
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(gray1, gray2, full=True) #SSIM score, (1=)
    return score

# MQTT callbacks 
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected successfully to the MQTT broker")
    else: 
        print(f"Failed to connect, return code {rc}")
        
# def on_publish(client, userdata, mid):
#     print(f"message {mid} published.")
    
def on_disconnect(client, userdata, rc, properties=None, reasoncode=None):
    if rc != 0:
        print("Disconnected unexpectedly from the MQTT broker. Trying to reconnect...")
        client.reconnect()
    else:
        print("Disconnected succesfully from MQTT")
        
# 
def recv_line(sock):
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("Socket closed while reading line.")
        data += chunk
    return data.decode("utf-8").strip()

# Receive data over wifi    
def receive_data(client_socket):
        try:
            # Receive label
            #label = client_socket.recv(1024).decode("utf-8").strip()
            label = recv_line(client_socket)
            print(f"Received label: {label}")
            
            # Receive prob.
            prob_bytes= client_socket.recv(4)
            probability = struct.unpack("f", prob_bytes)[0]
            print(f"Received probability: {probability}")
            
            # Receive image size 
            img_size_bytes = client_socket.recv(4)
            img_size = struct.unpack("!I", img_size_bytes)[0]
            print(f"Received image size: {img_size}")
            
            # Receive image data
            img_data = b""
            while len(img_data) < img_size:
                packet = client_socket.recv(1024)
                if not packet:
                    raise ConnectionError("Connection lost during image data reception")
                img_data += packet 
            print(f"Received image data: {len(img_data)} bytes")
                
            return label, probability, img_data
        
        except Exception as e:
            print(f"Error receiving data: {e}")
            
            # Flush bytes to realign stream
            try:
                client_socket.recv(1024)
            except:
                pass
            return None, None, None        
        
def connect_to_portenta():
    print(f"Starting server on {HOST}:{PORT}, waiting for the device to connect to it...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    #server_socket.settimeout(2) 
    
    while keep_running:
        try: 
            
            client_socket, addr = server_socket.accept()
            print(f"Connected to the device at {addr[0]}:{addr[1]}")
            return client_socket
        except socket.timeout:
            continue
    
    print("Server stopped")
    server_socket.close()
    return None
            
    
                
 
def send_data_mqtt(mqtt_client, label, probability, img_data, MQTT_TOPIC, mqtt_subtopic):
    
    try: 
        # Encode in base64
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # payload 
        payload = {
            "label" : label, 
            "probability": probability,
            "image": img_base64
        }
        
        # Publish metadata MQTT 
        mqtt_client.publish(MQTT_TOPIC + mqtt_subtopic, json.dumps(payload))
        print(f"Published to MQTT topic {MQTT_TOPIC}{mqtt_subtopic}")
        
        # # Publish jpeg MQTT
        # mqtt_client.publish(MQTT_TOPIC + "/image", img_data)
        # print(f"Image published to MQTT topic {MQTT_TOPIC}/image")
        
    except Exception as e:
        print(f"Error publishing data: {e}")


          

def start_client_portenta():
    global keep_running
    previous_image = None
    
    while keep_running:
        
        # Create new socket at each connection attempt
        client_socket = connect_to_portenta()
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE,1)
        if client_socket is None:
            keep_running = False
            break

        while keep_running:
            # Receive from Nicla 
            label, probability, img_data = receive_data(client_socket)
        
            if label is not None and img_data is not None: 
                
                # Decode image for cv
                current_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                if previous_image is not None: 
                    similarity_score = check_image_similarity(previous_image, current_image)
                    print(f"Image similarity score: {similarity_score})")
                    
                    if similarity_score >= similarity_threshold:
                        print("Images are too similar")
                        send_data_mqtt(mqtt_client, "Image too similar", "100%", img_data, MQTT_TOPIC, MQTT_SUBTOPIC2)
                    else:
                        send_data_mqtt(mqtt_client, label, probability, img_data, MQTT_TOPIC, MQTT_SUBTOPIC1)                                                         
                previous_image = current_image
            
            else: 
                print("Connection lost or error receiving data. Attempting to reconnect...")
                client_socket.close()
                break  
                
        if client_socket:  
            client_socket.close()
            print("Connection with Portenta closed")
            time.sleep(2)
                              

# Set up MQTT client
MQTT_BROKER = "localhost"
MQTT_PORT = 1883                      #Default port
mqtt_username = "username"
mqtt_password = "password"
mqtt_client_id = "Portenta_Publisher"
MQTT_TOPIC = "leaf"
MQTT_SUBTOPIC1 = "/prediction"
MQTT_SUBTOPIC2 = "/too_similar"

        
# Server config 
HOST = "0.0.0.0"
PORT = 8080

keep_running = True  
similarity_threshold = 0.9


if __name__ == "__main__":
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,mqtt_client_id)
    mqtt_client.username_pw_set(mqtt_username, mqtt_password)


    # MQTT callbacks 
    mqtt_client.on_connect = on_connect
    #mqtt_client.on_publish = on_publish
    mqtt_client.on_disconnect = on_disconnect


    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()

    except Exception as e:
        print(f"error connecting to MQTT broker: {e}")
        exit()

    # Client thread

    client_thread = threading.Thread(target= start_client_portenta)
    
    try:
        client_thread.start()
    
        while client_thread.is_alive():
            
            client_thread.join(timeout=1)
        
    except KeyboardInterrupt:
        print("Interrupted, stopping...")
        keep_running = False
        client_thread.join()
        
        
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("MQTT client disconnected.")

    