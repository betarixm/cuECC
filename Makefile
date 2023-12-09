PROJECT_DIR = $(realpath .)

BUILD_DIR = $(PROJECT_DIR)/build
SRC_DIR = $(PROJECT_DIR)/src

LIB_TARGET = $(BUILD_DIR)/libcuecc.so
LIB_SOURCE = $(SRC_DIR)/*.cu
LIB_DEPENDENCIES = $(SRC_DIR)/**/*.cuh

NVCC = nvcc
NVCC_FLAGS = -Xcompiler -fPIC -shared -rdc=true -o $(LIB_TARGET)

define COMPILE_COMMANDS 
[\
    {\
        \"directory\": \"$(BUILD_DIR)\",\
        \"command\": \"$(NVCC) $(LIB_SOURCE)\",\
        \"file\": \"$(LIB_SOURCE)\"\
    }\
]
endef

compile_commands.json: Makefile
	@echo $(COMPILE_COMMANDS) > compile_commands.json

$(LIB_TARGET): $(LIB_SOURCE) $(LIB_DEPENDENCIES)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(LIB_SOURCE)

all: $(LIB_TARGET)
