# Makefile Example to deploy TVM modules.
TVM_ROOT="/root/tvm"
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++14 -O2 -fPIC\
        -I${TVM_ROOT}/include\
        -I${DMLC_CORE}/include\
        -I${TVM_ROOT}/3rdparty/dlpack/include\

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

.PHONY: clean all

all: tvm_c_inference

# Build rule for all in one TVM package library
libtvm_runtime_pack.o: tvm_runtime_pack.cc
        $(CXX) -c $(PKG_CFLAGS) -o $@  $^

# Deploy using the all in one TVM package library
tvm_c_inference: tvm_c_inference.c libtvm_runtime_pack.o cnn_tvm_lib.so
        $(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

clean:
        rm -fr libtvm_runtime_pack.o tvm_c_inference
