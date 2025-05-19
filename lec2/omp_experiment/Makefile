CC = /opt/homebrew/opt/llvm/bin/clang
CFLAGS = -fopenmp -O2

all: my_program

my_program: experiment.c
	$(CC) $(CFLAGS) -o my_program experiment.c

run: my_program
	OMP_DISPLAY_ENV=VERBOSE ./my_program
