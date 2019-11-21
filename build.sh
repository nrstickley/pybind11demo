#! /bin/bash

g++ -Ofast -march=native -mfpmath=sse -fopenmp -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` pbdemo.cpp -o example`python3-config --extension-suffix`

strip *.so
