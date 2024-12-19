all: liu-sun_CPU.exe liu-sun_serial.exe

liu-sun_CPU.exe: liu-sun_CPU.cpp
	g++ -o liu-sun_CPU.exe -std=c++20 -O3 -Wall -Wconversion -Wextra liu-sun_CPU.cpp
liu-sun_serial.exe: liu-sun_serial.cpp
	g++ -o liu-sun_serial.exe -std=c++20 -O3 -Wall -Wconversion -Wextra liu-sun_serial.cpp

clean:
	rm -f liu-sun_CPU.exe liu-sun_serial.exe
