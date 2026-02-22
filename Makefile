CC = clang
UNAME_S := $(shell uname -s)

CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic -Wno-c11-extensions -Wno-missing-field-initializers -I vendor
LDFLAGS = -lm -lpthread

ifeq ($(UNAME_S),Darwin)
	CFLAGS += -D_DARWIN_C_SOURCE -Wno-typedef-redefinition
else
	CFLAGS += -D_GNU_SOURCE
	LDFLAGS += -ldl
endif

SRC_DIR = src
BUILD_DIR = build
VENDOR_DIR = vendor

SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/vectordb.c $(SRC_DIR)/memory.c $(SRC_DIR)/metadata.c $(SRC_DIR)/vulkan_backend.c $(VENDOR_DIR)/volk.c
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/memo
BINDIR ?= $(HOME)/.local/bin
SHADER_SRC = shaders/memo_search.comp
SHADER_SPV = $(BUILD_DIR)/memo_search.spv
LLM_SHADER_SRC = shaders/headless.comp
LLM_SHADER_SPV = $(BUILD_DIR)/headless.spv

.PHONY: all clean install uninstall

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

install: all
	@mkdir -p $(BINDIR)
	ln -sfn $(abspath $(TARGET)) $(BINDIR)/memo
	@echo "Installed memo symlink to $(BINDIR)/memo -> $(abspath $(TARGET))"
	@echo "Ensure $(BINDIR) is in PATH"
	@echo "Add to ~/.zshrc: export PATH=\"$(BINDIR):\$$PATH\""

uninstall:
	rm -f $(BINDIR)/memo
	@echo "Removed $(BINDIR)/memo"
