CC=gcc
BUILD=./build/
BIN=$(BUILD)test
LIBRARIES= -lm

all:
	mkdir -p build
	$(CC) main.cpp -o $(BIN) $(LIBRARIES)

clean: 
	rm -fr build
