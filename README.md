# MyDrone
- Turn on Fan with gpio398 (jetson TX2): https://forums.developer.nvidia.com/t/fan-not-spinning-gets-really-hot-auvidea-j120/50435/9
	$ sudo su
	# echo 398 > /sys/class/gpio/export
	# echo "out" > /sys/class/gpio/gpio398/direction
	# echo 1 > /sys/class/gpio/gpio398/value # Turn off FAN
	# echo 0 > /sys/class/gpio/gpio398/value # Turn on FAN
	# exit

- Install Tensorflow on Jetson TX2: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

- Real Sense Depth Camera D435i:
+ Download librealsense: https://github.com/IntelRealSense/librealsense
+ Install librealsense for Linux using Ubuntu: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
