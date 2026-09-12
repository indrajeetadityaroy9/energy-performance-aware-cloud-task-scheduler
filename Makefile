.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:

PROJECT := mcc_scheduler
SRC_DIR := src
TEST_DIR := tests
INCLUDE_DIR := include
SCRIPT_DIR := scripts

BUILD_DIR ?= build
BIN_DIR ?= bin
TARGET := $(BIN_DIR)/$(PROJECT)
TEST_TARGET := $(BIN_DIR)/mcc_tests

LIB_SOURCES := \
	$(SRC_DIR)/mcc.cpp \
	$(SRC_DIR)/examples.cpp \
	$(SRC_DIR)/experiments.cpp
APP_SOURCES := $(LIB_SOURCES) $(SRC_DIR)/main.cpp
TEST_SOURCES := $(LIB_SOURCES) $(TEST_DIR)/mcc_regression.cpp

APP_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(APP_SOURCES))
TEST_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TEST_SOURCES))
DEPENDENCIES := $(sort $(APP_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d))

CLI_TEST := $(TEST_DIR)/cli_tests.sh
EXPERIMENT_RUNNER := $(SCRIPT_DIR)/run_experiments.sh

ifeq ($(origin CXX),default)
CXX := c++
endif

CXX_STANDARD ?= c++17
WARNINGS ?= \
	-Wall \
	-Wextra \
	-Wpedantic \
	-Wshadow \
	-Wconversion \
	-Wsign-conversion
OPTIMIZATION ?= -O2
DEPENDENCY_FLAGS := -MMD -MP

EXTRA_CPPFLAGS ?=
EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS ?=
LDLIBS ?=

PROJECT_CPPFLAGS := -I$(INCLUDE_DIR) $(EXTRA_CPPFLAGS)
PROJECT_CXXFLAGS := \
	-std=$(CXX_STANDARD) \
	$(WARNINGS) \
	$(OPTIMIZATION) \
	$(DEPENDENCY_FLAGS) \
	$(EXTRA_CXXFLAGS)
PROJECT_LDFLAGS := $(EXTRA_LDFLAGS)

# Set V=1 to print complete compiler and linker commands.
ifeq ($(V),1)
Q :=
else
Q := @
endif

.PHONY: all
all: $(TARGET)

$(TARGET): $(APP_OBJECTS)
	@mkdir -p $(@D)
	@printf '  LINK    %s\n' $@
	$(Q)$(CXX) $(PROJECT_LDFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(TEST_TARGET): $(TEST_OBJECTS)
	@mkdir -p $(@D)
	@printf '  LINK    %s\n' $@
	$(Q)$(CXX) $(PROJECT_LDFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@printf '  CXX     %s\n' $<
	$(Q)$(CXX) $(CPPFLAGS) $(PROJECT_CPPFLAGS) $(PROJECT_CXXFLAGS) $(CXXFLAGS) \
		-c $< -o $@

.PHONY: run test check experiments
run: $(TARGET)
	$(Q)$(TARGET)

test: $(TEST_TARGET) $(TARGET)
	$(Q)$(TEST_TARGET)
	$(Q)sh $(CLI_TEST) $(TARGET)

check: test

experiments: $(TARGET)
	$(Q)sh $(EXPERIMENT_RUNNER) $(TARGET)

# Isolated configurations prevent objects compiled with incompatible flags
# from being silently reused across development workflows.
RELEASE_BUILD_DIR ?= $(BUILD_DIR)/release
RELEASE_BIN_DIR ?= $(BIN_DIR)/release
SANITIZE_BUILD_DIR ?= $(BUILD_DIR)/sanitize
SANITIZE_BIN_DIR ?= $(BIN_DIR)/sanitize
SANITIZERS ?= address,undefined
SANITIZER_FLAGS := -fsanitize=$(SANITIZERS) -fno-omit-frame-pointer

.PHONY: release sanitize

release:
	+$(Q)$(MAKE) \
		BUILD_DIR="$(RELEASE_BUILD_DIR)" \
		BIN_DIR="$(RELEASE_BIN_DIR)" \
		OPTIMIZATION=-O3 \
		EXTRA_CXXFLAGS="$(EXTRA_CXXFLAGS) -DNDEBUG" \
		all

sanitize:
	+$(Q)$(MAKE) \
		BUILD_DIR="$(SANITIZE_BUILD_DIR)" \
		BIN_DIR="$(SANITIZE_BIN_DIR)" \
		OPTIMIZATION=-O1 \
		EXTRA_CXXFLAGS="$(EXTRA_CXXFLAGS) -g3 $(SANITIZER_FLAGS)" \
		EXTRA_LDFLAGS="$(EXTRA_LDFLAGS) -fsanitize=$(SANITIZERS)" \
		test

.PHONY: clean info help
clean:
	@set -eu; \
	for directory in "$(BUILD_DIR)" "$(BIN_DIR)"; do \
		normalized=$$directory; \
		while [ "$${normalized%/}" != "$$normalized" ]; do \
			normalized=$${normalized%/}; \
		done; \
		case "$$normalized" in \
			''|.|..|/*|-*) \
				printf 'Refusing to remove unsafe directory: %s\n' "$$directory" >&2; \
				exit 2 ;; \
		esac; \
		case "/$$normalized/" in \
			*/../*) \
				printf 'Refusing to remove unsafe directory: %s\n' "$$directory" >&2; \
				exit 2 ;; \
		esac; \
	done; \
	printf '  CLEAN   %s %s\n' "$(BUILD_DIR)" "$(BIN_DIR)"; \
	$(RM) -r "$(BUILD_DIR)" "$(BIN_DIR)"

info:
	@printf 'CXX=%s\n' "$(CXX)"
	@printf 'CXX_STANDARD=%s\n' "$(CXX_STANDARD)"
	@printf 'BUILD_DIR=%s\n' "$(BUILD_DIR)"
	@printf 'BIN_DIR=%s\n' "$(BIN_DIR)"
	@printf 'CPPFLAGS=%s\n' "$(CPPFLAGS) $(PROJECT_CPPFLAGS)"
	@printf 'CXXFLAGS=%s\n' "$(PROJECT_CXXFLAGS) $(CXXFLAGS)"
	@printf 'LDFLAGS=%s\n' "$(PROJECT_LDFLAGS) $(LDFLAGS)"

help:
	@printf '%s\n' \
		'Targets:' \
		'  all         Build the scheduler (default)' \
		'  test        Run C++ regression and CLI acceptance tests' \
		'  run         Run the built-in paper examples' \
		'  experiments Run the complete Section IV-style sweep' \
		'  release     Build an isolated optimized release binary' \
		'  sanitize    Run tests with AddressSanitizer and UBSan' \
		'  info        Print the resolved build configuration' \
		'  clean       Remove artifacts for the selected directories' \
		'' \
		'Common overrides: CXX, CXX_STANDARD, BUILD_DIR, BIN_DIR, V=1'

-include $(DEPENDENCIES)
