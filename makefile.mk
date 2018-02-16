CC = g++
OPT = -O4 -std=c++11 -lm

AcceleratedGradientTest.exe : AcceleratedGradientTest.cpp manifold_opt/accel_grad_descent.h stiefel_retraction/cayley_retraction.h
	$(CC) -o AcceleratedGradientTest.exe AcceleratedGradientTest.cpp $(OPT)
