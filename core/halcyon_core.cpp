#include<iostream>
#include <pybind11/detail/common.h>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<string>
#include<cpp20template/utils.h>

void say_hello(){
    std::string msg = R"(
        halcyon_core is saying hello to you!
    )";
    utils::println(msg);
}


PYBIND11_MODULE(halcyon_core, m){
    m.def("say_hello", say_hello);
}
