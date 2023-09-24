CC=gcc
BUILD=./build/
BIN=$(BUILD)test
LIBRARIES= -lm -lstdc++

all:
	mkdir -p build
	$(CC) main.c  -ggdb -pedantic -Wall -o $(BIN) $(LIBRARIES)

run:
	make all
	./$(BIN)

d: 
	gdb ./build/test

clean: 
	rm -fr build
