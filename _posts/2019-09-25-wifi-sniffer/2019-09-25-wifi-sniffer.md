---
title: Build a WiFi Sniffer
description:
categories:
- Hello1
tags:
- Test1
- Test2
---

> Build a wifi sniffer

List of components:
* Arduino
* [ESP8266 Wifi module]()
* [Xbee]()

## Program ESP8266 module
We need to program the ESP8266 to be a wifi sniffer. The code needs to be uploaded to the module in Arduino IDE yet bypassing the Arduino itself (i.e. use Arduino as a ESP8266 board). To do this, follow the [Arduino core for ESP8266 wifi chip](https://github.com/esp8266/Arduino) to install via board manager in Arduino IDE. Remember to use version 2.5.0 instead of the latest version. Then:
1. Connect ESP8266 TX/RX, VCC, GND, GPIO0 (grounded to put in in flash mode) with Arduino except RESET. Power-on Arduino.
2. Wait for 5 sec for Arduino to boot (Important)
3. Ground RESET of Arduino to bypass it as a USB-Serial transceiver. It will disable Arduino Board and upload code directly to the ESP8266
4. Open Arduino IDE, "Tools-->Boards: Generic ESP8266 Board Module-->Upload Speed: 115200", upload the [wifi sniffer program](https://www.hackster.io/rayburne/esp8266-mini-sniff-f6b93a) to ESP8266
5. Should see TX on Arduino and blue light on ESP8266 blinking, indicating the program is being upload
6. toggle baud rate to 57600, serial monitor will show packet sniffed.

Be careful. There were many weird errors in this uploading step. For example, [Fatal Error: Failed to connect](https://github.com/espressif/esptool/issues/407) solved by downgrade esp8266 Arduino core to v2.5.0, [sync error](https://tttapa.github.io/Pages/Arduino/ESP8266/Flashing/Flashing-With-an-Arduino.html) solved by strictly following the above [steps](https://www.hackster.io/harshmangukiya/how-to-program-esp8266-with-arduino-uno-efb05f).

Upon successful uploading, the ESP8266 will be running in promiscuous mode which will display Device and Access Point Wifi 2.4G band MAC, RSSI (relative signal strength index), SSID, and channel. Since the signal strength is shown, you can even use this to move your Access Point (WiFi router) around to find the best placement location in your home.

Check `functions.h: promisc_cb()` for details on this callback function for promiscuous mode Wifi detection. In promiscuous mode, all WiFi traffic will be passed for processing instead of only accepting packets designated for it.

Explanation:
An Access Point, AP, will occasionally "beacon" so that WiFi devices and other APs will know of its existence. The RSSI is a signal indicator of the received power: generally speaking, the higher the absolute number the weaker the signal (due to the minus sign.) A MAC address of all "ffffffffffff" is called a broadcast which has special network significance.

On the client (device) side, a MAC will be printed followed by "==>" to indicate communication (typically) with an AP. Again, the MAC of the receiving device is shown as well as the channel and the signal level.

A follow-up of [probe request sniffer](https://www.hackster.io/kosme/esp8266-sniffer-9e4770).

## Unique and Active MAC addresses
The example code given by Ray Burnette displays a list of router and devices along with the MAC addresses and other statistics through the serial port. However, for the purpose of this lab, the only relevant data is the estimation of the number of unique devices in the room. Thus, we should edit the code in function.h, and add a data structure to keep track of a list of unique WiFi MAC addresses and their timestamps. The example code already keeps track of the list of unique MAC addresses, and you should modify it to have a timeout for devices which have not been active for an arbitrary time interval.

## Air Quality sensor
```c
int sensorValue;

void setup() {  
    Serial.begin(9600); // sets the serial port to 9600
}

void loop() {
    sensorValue = analogRead(0); // read A0 analog pin
    Serial.print("Air Quality = ");
    Serial.print(sensorValue, DEC); // prints the air quality value in decimal
    Serial.println(" PPM"); // unit is Parts per million (PPM), normal value around 50ppm, range 10-300 for NH3, 10-1000 for benzene, 10-300 for alcohol
    delay(100);
}

```
