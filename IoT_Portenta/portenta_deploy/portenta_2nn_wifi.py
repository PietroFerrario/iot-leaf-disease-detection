# Untitled - By: piefe - Tue May 27 2025

# Portenta Neural Network Inference with WiFi
# By: User
# Date: 2025-05-13

import sensor, time, ml, gc, uos, struct, network, socket
from pyb import LED

# WiFi Configuration
SSID = "iPhone di Pie"
PASSWORD = "ciaociao"
SERVER_IP = "172.20.10.8"
SERVER_PORT = 8080

# Leaf detection
LEAF_THRESHOLD = 0.50

# Connect to WiFi
print("Connecting to WiFi...")
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

while not wlan.isconnected():
    print("Connecting...")
    time.sleep(1)

print("Connected to WiFi, IP address:", wlan.ifconfig()[0])

# Socket Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))
print("Connected to server")

# Camera Initialization
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)
sensor.set_windowing((96, 96))
sensor.skip_frames(time = 2000)

# Model and Label Initialization
gc.collect()
print("Loading model for LEAVES detection...")
model_leaf =ml.Model("leaf_detection.tflite", load_to_fb=uos.stat('leaf_detection.tflite')[6] > (gc.mem_free() - (64*1024)))
labels_leaf = [line.rstrip() for line in open("labels_leaf.txt")]
print("Model for LEAVES detection loaded and labels read")

# Model for sickness detection
gc.collect()
print("Loading model for HEALTH detection...")
model_health =ml.Model("health_detection.tflite", load_to_fb=uos.stat('health_detection.tflite')[6] > (gc.mem_free() - (64*1024)))
labels_health = [line.rstrip() for line in open("labels_health.txt")]
print("Model for HEALTH detection loaded and labels read")


# LED Setup
led1 = LED(1)
led2 = LED(2)

def send_data(sock, label, probability, img):
    try:
        # Send label
        sock.sendall(label.encode('utf-8') + b"\n")
        #sock.sendall((label + "::END::").encode('utf-8'))

        # Send probability
        prob_bytes = struct.pack("f", probability)
        sock.sendall(prob_bytes)

        # Send image
        img_jpeg = img.to_jpeg(quality=35)
        img_size = len(img_jpeg)
        sock.sendall(struct.pack("!I", img_size))
        sock.sendall(img_jpeg)

        print(f"Sent: {label}, {probability}, {img_size} bytes")
    except Exception as e:
        print(f"Error sending data: {e}")

while True:
    try:
        led2.on()
        img = sensor.snapshot()
        led2.off()

        # Run inference for leaf detection
        predictions_list_leaf = list(zip(labels_leaf,  model_leaf.predict([img])[0].flatten().tolist()))
        for i in range(len(predictions_list_leaf)):
            print("%s = %f" % (predictions_list_leaf[i][0], predictions_list_leaf[i][1]))

        leaf_prob = next(pred for label, pred in predictions_list_leaf if label == "leaves")

        # If it's a leaf
        if leaf_prob >= LEAF_THRESHOLD:

            # Run inference for health detection
            predictions_list_health = list(zip(labels_health,  model_health.predict([img])[0].flatten().tolist()))
            for i in range(len(predictions_list_health)):
                print("%s = %f" % (predictions_list_health[i][0], predictions_list_health[i][1]))
            most_probable_health = max(predictions_list_health, key = lambda item: item[1])

            # Send data
            send_data(sock, most_probable_health[0], most_probable_health[1]*100, img)

            time.sleep(3)

        else:

            print("Image is not a leaf! Point the camera toward a leaf!")

            time.sleep(1)

    except Exception as e:
        print(f"Main loop error: {e}")
        time.sleep(5)
