#L serial/Makefile
# Builds lda_serial.so shared library

CC = gcc
CFLAGS = -O2 -Wall -fPIC -std=c99
LDFLAGS = -shared -lm

TARGET = lda_serial.so
SRCS = lda_serial.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
