# Minimal Makefile (MSYS2/mingw64 + OpenCV via pkg-config)

CXX       := g++
CXXFLAGS  := -std=gnu++17 -O2 -Wall -Wextra -Wno-unused-parameter \
             $(shell pkg-config --cflags opencv4)
LDFLAGS   := $(shell pkg-config --libs opencv4)

BIN_DIR   := bin
TARGET    := $(BIN_DIR)/app.exe

SOURCES   := $(wildcard src/*.cpp) $(wildcard *.cpp)
OBJECTS   := $(patsubst src/%.cpp,$(BIN_DIR)/%.o,$(filter src/%.cpp,$(SOURCES))) \
             $(patsubst %.cpp,$(BIN_DIR)/%.o,$(filter %.cpp,$(SOURCES)))
DEPS      := $(OBJECTS:.o=.d)

.PHONY: all clean run
all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/%.o: src/%.cpp | $(BIN_DIR)
	@$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/%.o: %.cpp | $(BIN_DIR)
	@$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	@$(TARGET)

clean:
	@rm -f $(OBJECTS) $(DEPS) $(TARGET)

-include $(DEPS)
