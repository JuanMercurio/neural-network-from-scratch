CC=gcc
BUILD=./build/
BIN=$(BUILD)test

all:
	$(CC) main.cpp -o $(BIN)
