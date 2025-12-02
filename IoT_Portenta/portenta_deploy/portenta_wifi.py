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
print("Loading model...")
model =ml.Model("health_detection.tflite", load_to_fb=uos.stat('health_detection.tflite')[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip() for line in open("labels_health.txt")]
print("Model loaded and labels read")

# LED Setup
led1 = LED(1)
led2 = LED(2)

def send_data(sock, label, probability, img):
    try:
        # Send label
        sock.sendall(label.encode('utf-8') + b"\n")

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

        # Run inference
        print("Raw predictions:", model.predict([img])[0].flatten().tolist())
        predictions_list = list(zip(labels,  model.predict([img])[0].flatten().tolist()))
        for i in range(len(predictions_list)):
            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))
        most_probable = max(predictions_list, key = lambda item: item[1])

        # Send data
        print("Label to send:", most_probable[0])
        print("Probability to send.", most_probable[1])
        send_data(sock, most_probable[0], most_probable[1]*100, img)

        time.sleep(3)

    except Exception as e:
        print(f"Main loop error: {e}")
        time.sleep(5)
