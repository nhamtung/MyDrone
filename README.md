# MyDrone
- Turn on Fan with gpio398 (jetson TX2): https://forums.developer.nvidia.com/t/fan-not-spinning-gets-really-hot-auvidea-j120/50435/9
	$ sudo su
	# echo 398 > /sys/class/gpio/export
	# echo "out" > /sys/class/gpio/gpio398/direction
	# echo 1 > /sys/class/gpio/gpio219/value # Turn off FAN
	# echo 0 > /sys/class/gpio/gpio219/value # Turn on FAN
	# exit
