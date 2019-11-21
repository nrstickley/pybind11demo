#include <iostream>
#include <algorithm>

#include <omp.h>
#include <immintrin.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


/** An example function */
int add(int i, int j) {
    return i + j;
}


/** Passing a NumPy array as an argument and returning a float.
 *  This is the straightforward single-threaded version of the function
 *  with the attribute hot added to help the compiler. Notice that the 
 *  sum is accumulated in a 32-bit float, so some precision is lost.
*/
__attribute__ ((hot)) float numpy_sum_st(const py::array_t<float, py::array::c_style> &array)
{
    float sum = 0.0;

    const float* data = array.data();

    const unsigned int size = array.size();

    for (unsigned int i=0; i < size; ++i)
    {
        sum += *data;
        ++data;
    }

    return sum;
}


/** The same as above, but now explicitly using AVX2, but not handling edge cases.*/
__attribute__ ((hot)) float numpy_sum_avx(const py::array_t<float, py::array::c_style> &array)
{
    __m256 sum = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    const float* data = array.data();

    const unsigned int size = array.size();

    for (unsigned int i=0; i < size; i+=8)
    {
        const __m256 octet = _mm256_load_ps(data);
        sum = _mm256_add_ps(sum, octet);
        data += 8;
    }

    union { __m256 vector; float array[8]; } val;

    val.vector = sum;

    return val.array[0] + val.array[1] + val.array[2] + val.array[3] 
         + val.array[4] + val.array[5] + val.array[6] + val.array[7];
}


/** Similar to the cases above, but now using multithreading. */
float numpy_sum_mt(const py::array_t<float, py::array::c_style> &array)
{
    float sum = 0.0;

    const float* data = array.data();

    const unsigned int size = array.size();

    #pragma omp parallel for reduction(+ : sum) schedule(static) num_threads(8)
    for (unsigned int i=0; i < size; ++i)
    {
        sum += data[i];
    }

    return sum;
}

/** This is the fuction that I will actually wrap. It's adaptive, calling the single-threaded
 *  version of the sum function for short arrays and the OpenMP version for larger arrays. 
*/
float numpy_sum(const py::array_t<float, py::array::c_style> &array)
{
    const unsigned int size = array.size();

    float result = 0.0f;

    if (size < 40000u)
    {
        result = numpy_sum_st(array);
    }
    else
    {
        result = numpy_sum_mt(array);
    }

    return result;
}


/** Similar to the sum function above, but now finding the maximum value of an array*/
float numpy_max(py::array_t<float, py::array::c_style> &array)
{
    float max = -INFINITY;

    const float* data = array.data();

    const unsigned int size = array.size();

    for (unsigned int i=0; i < size; ++i)
    {
        //max = std::max(array.at(i), max);
        max = std::max(*data, max);
        ++data;
    }

    return max;
}


/** An example class containing public and protected data and a public enumeration */
class Pet
{
public:
    enum Class {Mammal = 0 , Reptile, Bird, Fish, Arachnid, Insect};

    Pet(const std::string &name) : name(name) { }

    void set_name(const std::string &name_) { name = name_; }

    const std::string& get_name() const { return name; }

    void set_age(const float age) { _age = age; }

    float get_age() const { return _age; }

    void set_species(const std::string &species) {
        std::cout << "We are in the setter for species. We can perform some checks here.\n";
        _species = species;
    }

    const std::string& get_species() const { return _species; }

    std::string name;

    float mass = 0;

    int n_years = 0;

protected:

    float _age = 0;

    std::string _species = "";
};


PYBIND11_MODULE(example, m) {

    // module docstring

    m.doc() = "pybind11 example plugin";

    // expose a C++ function, name its parameters, and give it default parameters

    m.def("add", &add, "A function which adds two numbers",
          py::arg("i")=0, // default argument
          py::arg("j")=0
          );

    m.def("numpy_sum", &numpy_sum, "A function for summing a NumPy array.",
          py::arg("array"));

    m.def("numpy_max", &numpy_max, "A function for finding the value of the largest element of a NumPy array.",
          py::arg("array"));

    // cast a C++ type to a Python type and store it as a Python object

    py::object world = py::cast("World");

    // add attributes to the module

    m.attr("the_answer") = 42;
    m.attr("what") = world;

    // expose the Pet class

    py::class_<Pet> pet(m, "Pet");

    pet.def(py::init<const std::string &>())
       .def("set_name", &Pet::set_name, "docstring for set_name")
       .def("get_name", &Pet::get_name, "docstring for get_name")
       .def("set_age", &Pet::set_age, "docstring for set_age")
       .def("get_age", &Pet::get_age, "docstring for get_age")
       .def_readwrite("mass", &Pet::mass)
       .def_property("species", &Pet::get_species, &Pet::set_species, "docstring for species")
       .def("__str__", [](const Pet &p) { return "<a Pet named " + p.get_name() + ">"; });

    // add the enumeration to the Pet class

    py::enum_<Pet::Class>(pet, "Class", py::arithmetic())
        .value("Mammal",  Pet::Class::Mammal)
        .value("Reptile", Pet::Class::Reptile)
        .value("Bird",    Pet::Class::Bird)
        .value("Fish",    Pet::Class::Fish)
        .value("Arachnid", Pet::Class::Arachnid)
        .value("Insect",  Pet::Class::Insect)
        .export_values();
}
