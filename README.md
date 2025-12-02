# Real-Time Detection of Grapevine Leaf Diseases with Edge AI and IoT Technologies

## Abstract
Grapevine diseases and pests pose significant challenges to agricultural productivity, particularly in organic farming, which relies heavily on early detec- tion. Traditional methods of disease detection depends on manual inspection, which is labor-intensive and prone to errors. This project leverages advancements in edge AI and IoT technologies to develop a real-time system for the classification of the health status of grapevine leaf. The Nicla Vision was utilized as edge device to perform on-device inference using a quantized TensorFlow Lite model, achieving an accuracy of 93%. Blob detection was implemented to pre-process input images, ensuring efficient region identification and classification. The system transmits classification results, which include labels, prob- abilities and respective images, via an MQTT broker to a centralized data management pipeline, comprising Node- RED, InfluxDB and Grafana. This framework enables efficient data storage and visualization, making it highly adaptable for vineyard monitoring.

Below a figure with the complete pipeline of the system: 
![alt text](https://github.com/PietroFerrario/iot-leaf-disease-detection/blob/main/Figures/Diagramma.png)


## File structure
-Nicla Vision 
    - `Code_nicla/`: PC-side scripts to run with the NiclaVision board connected. Run: main.py
    - `nicla_deploy/`: Code deployed on the Nicla Vision 

- Portenta H7
    - `Code_portenta/`: PC-side scripts to run with the Portenta board connected. Run: portenta_wifi_laptop.py
    - `portenta_deploy/`: Code deployed on the Portenta H7. portenta_2nn_wifi.py for the dual neural-network setup.

- `Tensorflow/`: Full neural network pipeline: model definition, training, pruning, quantization, and fine-tuning.
