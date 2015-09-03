NVCC = nvcc
CUBITS = thresh.cu elemwise_math.cu
HEADERS = src/hnn_cubits.h
VPATH = src/
LIB_DIR = lib/
INSTALL_LIB_DIR = /usr/lib/
INSTALL_INCLUDE_DIR = /usr/include/
NVCC_FLAGS = --shared --compiler-options -fPIC

$(LIB_DIR)libhnn_cubits.so: $(CUBITS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

install: $(LIB_DIR)libhnn_cubits.so
	cp $(LIB_DIR)libhnn_cubits.so $(INSTALL_LIB_DIR)
	cp $(HEADERS) $(INSTALL_INCLUDE_DIR)

