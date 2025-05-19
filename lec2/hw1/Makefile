CC = /opt/homebrew/opt/llvm/bin/clang++
CFLAGS = -fopenmp -O2 -stdlib=libc++ -L$(shell brew --prefix llvm)/lib

all: my_program

my_program: lab1_v0.cpp
	$(CC) $(CFLAGS) -o my_program lab1_v0.cpp

run: my_program
	OMP_DISPLAY_ENV=VERBOSE ./my_program