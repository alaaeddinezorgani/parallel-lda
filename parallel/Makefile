#L parallel/Makefile
# Builds lda_openmp.so shared library with OpenMP support

CC = gcc
CFLAGS = -O3 -march=native -Wall -fPIC -fopenmp -std=c99
LDFLAGS = -shared -fopenmp -lm

TARGET = lda_openmp.so
SRCS = lda_openmp.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)


