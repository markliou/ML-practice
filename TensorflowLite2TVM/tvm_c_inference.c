#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

int main(){
        tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("cnn_tvm_lib.so");
}
