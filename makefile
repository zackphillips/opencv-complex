PROJECT = cMat.a
OBJECTS = cMat.o

LIBS =

# Extensions to clean
CLEANEXTS = o a

INCLUDES = -I/usr/local/include/ -I/usr/local/opt/opencv3/include -I/opt/local/include
# define the C compiler to use
CC = clang++

# define any compile-time flags
CFLAGS= -std=c++14 -ggdb -Wall -pedantic

LFLAGS=

all: $(PROJECT)

.cpp.o:
	$(CC) -c $(CFLAGS) $(INCLUDES) $< $(LFLAGS) $(LIBS)

$(PROJECT): $(OBJECTS)
	ar ru $@ $^

.PHONY: clean install
clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done

install:
	cp *.a /usr/local/lib/
	cp *.h /usr/local/include/
