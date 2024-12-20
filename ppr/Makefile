cuda1: cuda1.cu
	nvcc cuda1.cu -std=c++17 -O3 -o cuda1.exe -Dzisk -arch=sm_89

cpu1: cpu1.cpp
	g++ cpu1.cpp -std=c++17 -O3 -o cpu1.exe -Dzisk -fopenmp

serial1: serial1.cpp
	g++ serial1.cpp -std=c++17 -O3 -o serial1.exe -Dzisk

clean:
	rm -rf gpu1.exe cuda1.exe serial1.exe
