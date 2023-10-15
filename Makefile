CC=gcc
BUILD=./build/
BIN=$(BUILD)test
LIBRARIES= -lm -lstdc++
FLAGS= -ggdb -pedantic -Wall
C_FILES= $(shell find . -type f -name "*.c" | tr "\n" " ")

all:
	mkdir -p build
	$(CC) $(C_FILES) $(FLAGS) -o $(BIN) $(LIBRARIES)

run:
	make all
	./$(BIN)

d: 
	gdb ./build/test

v: all
	valgrind  --track-origins=yes --leak-check=full --log-file="valgrind.log" $(BIN)

clean: 
	rm -fr build *.log

print:
	echo $(C_FILES)

