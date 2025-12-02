# Untitled - By: piefe - Mon Sep 16 2024

#import sensor, image, time, ml, pyb, gc, uos, struct
#import socket
from lora import Lora

# Lora config
lora = Lora(debug=True)

dev_addr = "260B1B23"
nwk_skey = "CEB84244A129779402E9DD3496EB6F5E"
app_skey = "ECA7067D21F0AB59F1E3A5DFB1FB6DBF"


print("Attempting to join ABP")
if lora.join_ABP(None, dev_addr, nwk_skey, app_skey):
    print("LoraWAN with APB connection succesfull")
else:
    print("Join failed")

msg = "Test123"
print(f"Sending msg: {msg}")
lora.send_data(msg.encode(), confirmed=False)
print("Data Sent")
