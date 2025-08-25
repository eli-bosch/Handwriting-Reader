# Makefile for MSYS2/mingw64 + OpenCV via pkg-config
# Works with sources in src/*.cpp and/or *.cpp at project root.
#Thank you, ChatGPT for creating this makefile

# --- toolchain ---
CXX       := g++
CXXSTD    := -std=gnu++17
WARN      := -Wall -Wextra -Wno-unused-parameter
OPT       := -O2
PKGCONFIG := pkg-config

# --- OpenCV flags from pkg-config ---
OPENCV_CFLAGS := $(shell $(PKGCONFIG) --cflags opencv4)
OPENCV_LIBS   := $(shell $(PKGCONFIG) --libs opencv4)

# --- layout ---
BIN_DIR  := bin
TARGET   := $(BIN_DIR)/app.exe

# Find sources in src/ and project root
SOURCES_SRC := $(wildcard src/*.cpp)
SOURCES_ROOT := $(wildcard *.cpp)
SOURCES := $(SOURCES_SRC) $(SOURCES_ROOT)

# Guard: error out if no sources found
ifeq ($(strip $(SOURCES)),)
$(error No .cpp files found in src/ or project root. Add a file with int main().)
endif

# Map sources to bin/*.o (preserve only the basename)
OBJECTS_SRC  := $(patsubst src/%.cpp,$(BIN_DIR)/%.o,$(SOURCES_SRC))
OBJECTS_ROOT := $(patsubst %.cpp,$(BIN_DIR)/%.o,$(SOURCES_ROOT))
OBJECTS := $(OBJECTS_SRC) $(OBJECTS_ROOT)

.PHONY: all clean run show
all: $(TARGET)

# Link
$(TARGET): $(BIN_DIR) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(OPENCV_LIBS)

# Compile (sources under src/)
$(BIN_DIR)/%.o: src/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXSTD) $(WARN) $(OPT) $(OPENCV_CFLAGS) -c $< -o $@

# Compile (sources at project root)
$(BIN_DIR)/%.o: %.cpp | $(BIN_DIR)
	$(CXX) $(CXXSTD) $(WARN) $(OPT) $(OPENCV_CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	-@rm -f $(BIN_DIR)/*.o $(TARGET)

# Debug helper
show:
	@echo "SOURCES = $(SOURCES)"
	@echo "OBJECTS = $(OBJECTS)"
	@echo "OPENCV_CFLAGS = $(OPENCV_CFLAGS)"
	@echo "OPENCV_LIBS = $(OPENCV_LIBS)"
