//enable HNSW class to be seen by python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/hnsw.h"

namespace py = pybind11;

PYBIND11_MODULE(hnsw_module, m) {
    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int, int>(), 
             py::arg("max_layers") = 16, 
             py::arg("M") = 5, 
             py::arg("ef_construction") = 200)
        .def("insert", &HNSW::insert)
        .def("search", &HNSW::search);
}