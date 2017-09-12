C=gcc 
CFLAGS=-g -I. -I../src -std=c99 -Wall 
DEPS= ../src/art.h 
OBJ=art.o my_test.o

#%.o: %.c $(DEPS)
#	$(CC) -c -o $@ $< $(CFLAGS)

my_test: $(OBJ)
	$C -o my_test $(OBJ) $(CFLAGS) 

art.o: ../src/art.c
	$C -c ../src/art.c $(CFLAGS) 
my_test.o: my_test.c
	$C -c my_test.c $(CFLAGS) 

#my_test: $(OBJ)
#	gcc -o $@ $^ $(CFLAGS)
