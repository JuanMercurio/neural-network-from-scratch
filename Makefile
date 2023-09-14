CC=gcc
BUILD=./build/
BIN=$(BUILD)test
LIBRARIES= -lm

all:
	mkdir -p build
	$(CC) main.c -o $(BIN) $(LIBRARIES)

run:
	make all
	./$(BIN)

clean: 
	rm -fr build
