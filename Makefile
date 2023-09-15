CC=gcc
BUILD=./build/
BIN=$(BUILD)test
LIBRARIES= -lm -lstdc++

all:
	mkdir -p build
	$(CC) main.cpp -o $(BIN) $(LIBRARIES)

run:
	make all
	./$(BIN)

clean: 
	rm -fr build
