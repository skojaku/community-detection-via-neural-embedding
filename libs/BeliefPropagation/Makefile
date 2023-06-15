CC=g++
#CFLAGS= -O3 -pg -Wall
MYBINDIR=./belief_propagation
UNAME= $(shell uname)
ifeq ($(UNAME), Linux)
	CFLAGS=-O3
endif
ifeq ($(UNAME), Darwin)
	CFLAGS=-fast
endif
OFILE := ${MYBINDIR}/bm.o
CPPFILE := ${MYBINDIR}/bm.cpp
OUTPUT := ${MYBINDIR}/sbm
all: ${OUTPUT} ${OFILE}

${OUTPUT}: ${OFILE}
	${CC} -o $@ ${CFLAGS} $@.cpp ${OFILE}
${OFILE} : ${CPPFILE}
	${CC} ${CFLAGS} -c $< -o $@
clean:
	rm -f ${MYBINDIR}/*.o ${MYBINDIR}/sbm tags
