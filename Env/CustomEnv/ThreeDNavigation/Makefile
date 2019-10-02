CC = gcc
CXX = g++

HOME=/home/yangyutu/
VPATH = cppTest

DEBUGFLAG=-DDEBUG -g3 -O0 -fPIC
RELEASEFLAG= -O3 -march=native -DARMA_NO_DEBUG
CXXFLAGS=  -std=c++11 $(BOOST_INCLUDE) -D__LINUX  `python-config --cflags` `/home/yangyutu/anaconda3/bin/python -m pybind11 --includes` 
LDFLAG= -L/opt/OpenBLAS/lib  -pthread  `python-config --ldflags`

OBJ=main.o ActiveParticle3DSimulator.o

test.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 
	
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(DEBUGFLAG) $^
	




clean:
	rm *.o *.exe
	
