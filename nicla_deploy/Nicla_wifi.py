# Untitled - By: piefe - Mon Sep 16 2024

import sensor, image, time, ml, pyb, gc, uos, struct
import network
import socket

# Wifi-conf
SSID = "Nicla_AP"
KEY = "1234567890"
HOST = ""
PORT = 8080

# Initialize Access Point
wlan = network.WLAN(network.AP_IF)
wlan.config(ssid=SSID, password=KEY, channel=2)
wlan.active(True)

print("AP mode started. SSID: {} IP: {}".format(SSID, wlan.ifconfig()[0]))

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
server.bind([HOST, PORT])
server.listen(1)
# Set server socket to blocking
server.setblocking(True)

def start_streaming(server):
    print("Waiting for connections..")
    client, addr = server.accept()
    print("Connected to " + addr[0] + ":" + str(addr[1]))
    return client

# Initialize Camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000)

# Free memory before loading the model
gc.collect()
print("Loading model ...")
model =ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip("\n") for line in open("labels.txt")]
print("Model loaded and labels read")

threshold_green_leaf = (10, 90, -128, -10, -58, 50)

led1 = pyb.LED(2)
led1.off()
led2 = pyb.LED(3)
led2.off()

def reconnect_to_server():
    time.sleep(2)
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("Reconnection successful")
        return client_socket
    except Exception as e:
        print(f"Failed to reconnect: {e}")
        return None


def send_data(client_socket, label, probability, img):
    print(f"Leaf status at sending data: {label} with probability: {probability:.2f}")
    try:
        client_socket.sendall(label.encode('utf-8') + b"\n")
        print(f"Sent label: {label}")

        prob_bytes = struct.pack("f", probability)
        client_socket.sendall(prob_bytes)
        print(f"Sent probability: {probability:.2f}")

        img_jpeg = img.to_jpeg(quality=35, copy=False)
        img_size = len(img_jpeg)
        img_size_byte = struct.pack("!I", img_size)
        client_socket.sendall(img_size_byte)
        print(f"Sent image size: {img_size}")
        client_socket.sendall(img_jpeg)
        print(f"Sent image as jpeg, size: {len(img_jpeg)}")

    except OSError as e:
        print(f"Error sending data: {e}")

        if e.errno == 9:  #EBADf, socket close
            client_socket.close()
            client_socket = reconnect_to_server()

            if client_socket:
                print("Reconnected")
            else:
                print("Failed to reconnect, retrying...")
                client_socket = None
                global client
                client = None

    return client_socket


clock = time.clock()
last_detection_time = time.ticks_ms()
client = None

while True:
    try:
        if client is None:
            client = start_streaming(server)


        led2.on()
        clock.tick()

        current_time = time.ticks_ms()
        if time.ticks_diff(current_time, last_detection_time) > 2000:
            last_detection_time = current_time

            # Take a picture and brighten it
            #img = sensor.snapshot().gamma_corr(gamma=0.8, contrast=1.5)
            img = sensor.snapshot().gamma_corr(contrast=1.2)

            green_blobs = img.find_blobs([threshold_green_leaf], area_threshold=4000, merge=True)

            if green_blobs:
                print("Leaf-like object identified!")

                blob = max(green_blobs, key = lambda b: b.area())

                # Filter out images too big
                if blob.w() < 240 and blob.h() < 220:

                    img.draw_rectangle(blob.rect(), color=(255,0,0))
                    img.draw_cross(blob.cx(), blob.cy(), color=(255,0,0))
                    print(f"Blob detected at ({blob.x()}, {blob.y()}) with size ({blob.w()}x{blob.h()})")

                    # Flash the LED
                    led1.on()
                    time.sleep_ms(100)
                    led1.off()


                    # Resize Image
                    leaf_roi = (blob.x(), blob.y(), blob.w(), blob.h())
                    cropped_leaf = img.copy(roi=leaf_roi, hint=image.BILINEAR)

                    print("Cropped leaf size: {}x{}".format(cropped_leaf.width(), cropped_leaf.height()))

                    predictions_list = list(zip(labels, model.predict([cropped_leaf])[0].flatten().tolist()))
                    for i in range(len(predictions_list)):
                        print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

                    most_probable = max(predictions_list, key = lambda item: item[1])

                    send_data(client, most_probable[0], most_probable[1]*100, cropped_leaf)


                else:
                    send_data(client, "Error: Leaf too close to camera", 100, img)


            print(clock.fps())

    except OSError as e:
        print("Server socket error: ", e)
        if client:
            client.close()
            client = None
