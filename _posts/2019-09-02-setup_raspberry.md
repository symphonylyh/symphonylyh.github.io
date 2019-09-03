---
title: Setup an Raspberry Pi
description:
categories:
- Hello1
tags:
- Test1
- Test2
---

> We'll build a computer vision system for real-time object detection leveraging Raspberry Pi and camera. More realistic features such as distance sensing and vehicle modeling integrated with Arduino is yet to come.

List of components:
* [Raspberry Pi (3 Model B+)](https://www.amazon.com/gp/product/B07BDR5PDW/ref=ox_sc_act_title_11?smid=AAU5UPIIBDRLP&psc=1)
* [1080P Night-Vision Camera with two infrared LEDs](https://www.amazon.com/gp/product/B073183KYK/ref=ox_sc_act_title_10?smid=AMIHZKLK542FQ&psc=1)
* Power supply. Either a 5.1V-2.1A iPad charger or a power bank,  plus a USB to microUSB cable. Note that Raspberry Pi works normally at 5V-2.5A via microUSB.
* MicroSD (TF) card, recommend 32GB SDXC e.g. SanDisk Extreme Pro
* Keyboard, Mouse, Monitor, Ethernet cable, Ethernet switch, Raspberry fan, etc.

## Burn Raspbian OS
[PiBakery](https://www.pibakery.org/) is the most convenient way to quickly start Raspbian OS. You can customize as many steps as you want in PiBakery and they'll all run automatically when Pi is power-on. To simplify, we can just drag a "On First Boot" and a WiFi setup, and write to your microSD card (this takes a while). Now when you plug in the card, and the power is on, you'll see Raspbian OS booting right away.

## Configure Developer Environment
You have your tiny but angry Linux now, and you can do almost everything at your will.

### SSH
Following [Official Guide](https://www.raspberrypi.org/documentation/remote-access/ssh/), first turn on the SSH option. There is a low-level configuration panel we may use very often in the future:
```sh
sudo raspi-config
Go to 'Interface Options'
Select 'SSH', Yes-Ok-Finish
```

Get the IP address
```sh
ifconfig
or
hostname -I
```
Now you can SSH to it from Unix systems via
```sh
ssh@xxx.xxx.x.x
```
TODO: sshfs, ssh-keygen, etc

### Virtual Environment
```sh
python3 -m venv iot
source iot/bin/activate
deactivate
```

### Camera and CV modules
```sh
pip install --user picamera[array] # control camera via numpy
# it takes a while when building the numpy wheel
```
Now you test the camera by

```sh
raspistill -o cam.jpg
```
It will activate the camera and take a snapshot (there is a little delay).

```sh
# OpenCV
sudo apt update
sudo apt install libatlas3-base libsz2 libharfbuzz0b libtiff5 libjasper1 libilmbase12 libopenexr22 libilmbase12 libgstreamer1.0-0 libavcodec57 libavformat57 libavutil55 libswscale4 libqtgui4 libqt4-test libqtcore4 libwebp6 libhdf5-100
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
sudo pip3 install opencv-contrib-python

# TensorFlow
sudo apt install libatlas-base-dev
sudo pip3 install tensorflow
```

Download [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), ssdlite_mobilenet_v2_coco is a light-weight one.
