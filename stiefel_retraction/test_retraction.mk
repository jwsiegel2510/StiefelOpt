CC = g++
OPT = -std=c++11 -lm

test : retraction_test.exe
	./retraction_test.exe

retraction_test.exe : retraction_test.cpp cayley_retraction.h
	$(CC) -o retraction_test.exe retraction_test.cpp $(OPT)
