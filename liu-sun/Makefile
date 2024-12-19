all: liu-sun_CPU liu-sun_serial

liu-sun_CPU: liu-sun_CPU.cpp
	g++ -o liu-sun_CPU -std=c++20 -O3 -Wall -Wconversion -Wextra liu-sun_CPU.cpp
liu-sun_serial: liu-sun_serial.cpp
	g++ -o liu-sun_serial -std=c++20 -O3 -Wall -Wconversion -Wextra liu-sun_serial.cpp

clean:
	rm -f liu-sun_CPU liu-sun_serial