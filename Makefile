CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Wshadow -O2 -MMD -MP
SRC_DIR := src/cpp
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
TARGET := $(BIN_DIR)/mcc_scheduler
SOURCES := $(SRC_DIR)/mcc.cpp
OBJECTS := $(BUILD_DIR)/mcc.o
DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET)

-include $(DEPS)
