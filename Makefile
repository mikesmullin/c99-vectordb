CC = clang
CFLAGS = -std=c99 -O3 -Wall -Wextra -pedantic -D_GNU_SOURCE
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build

SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/vectordb.c $(SRC_DIR)/memory.c
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/vdb

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
