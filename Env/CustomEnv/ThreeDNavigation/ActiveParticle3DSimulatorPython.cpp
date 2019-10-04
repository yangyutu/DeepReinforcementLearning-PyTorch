
/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11' ]
cfg['linker_args'] = ['-L/opt/OpenBLAS/lib  -llapack -lblas  -pthread -no-pie']
cfg['include_dirs']= ['-I/home-4/yyang60@jhu.edu/work/Yang/Downloads/json/include']
cfg['sources'] = ['ActiveParticle3DSimulator.cpp']
%>
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "ActiveParticle3DSimulator.h"
namespace py = pybind11;


PYBIND11_MODULE(ActiveParticle3DSimulatorPython, m) {
    py::class_<ActiveParticle3DSimulator>(m, "ActiveParticle3DSimulatorPython")
        .def(py::init<std::string, int>())
        .def("createInitialState", &ActiveParticle3DSimulator::createInitialState)
        .def("setInitialState", &ActiveParticle3DSimulator::setInitialState)
        .def("step", &ActiveParticle3DSimulator::step)
        .def("getLocalFrame", &ActiveParticle3DSimulator::get_particle_local_frame)
    	.def("getPositions", &ActiveParticle3DSimulator::get_positions);
}
