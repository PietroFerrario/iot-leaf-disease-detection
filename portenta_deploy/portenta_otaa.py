from lora import Lora

# Lora config
lora = Lora(debug=True)

# OTAA Parameters
app_eui = "0000000000000000"
dev_eui = "70B3D57ED006FEEF"
app_key = "FDFAD3F40486392D33226870FA8DA4B0"

print("Attempting to join OTAA")
if lora.join_OTAA(app_eui, app_key):
    print("LoRaWAN OTAA connection successful")
else:
    print("Join failed")

msg = "Test123"
print(f"Sending msg: {msg}")
lora.send_data(msg.encode(), confirmed=False)
print("Data Sent")
