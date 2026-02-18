CC = clang
CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic -D_GNU_SOURCE -I vendor
LDFLAGS = -lm -ldl -lpthread

SRC_DIR = src
BUILD_DIR = build
VENDOR_DIR = vendor

SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/vectordb.c $(SRC_DIR)/memory.c $(SRC_DIR)/vulkan_backend.c $(VENDOR_DIR)/volk.c
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/memo
SHADER_SRC = shaders/memo_search.comp
SHADER_SPV = $(BUILD_DIR)/memo_search.spv
LLM_SHADER_SRC = shaders/headless.comp
LLM_SHADER_SPV = $(BUILD_DIR)/headless.spv

.PHONY: all clean

all: $(TARGET) $(SHADER_SPV) $(LLM_SHADER_SPV)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(SHADER_SPV): $(SHADER_SRC)
	@mkdir -p $(BUILD_DIR)
	glslc $< -o $@

$(LLM_SHADER_SPV): $(LLM_SHADER_SRC)
	@mkdir -p $(BUILD_DIR)
	glslc $< -o $@

clean:
	rm -rf $(BUILD_DIR)
