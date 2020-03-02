CXX=g++ -m64
CFLAGS=-fPIC


ifdef OS
   RM = del /Q
   FixPath = $(subst /,\,$1)
else
   ifeq ($(shell uname), Linux)
      RM = rm -f
      FixPath = $1
   endif
endif

all: libcgco.dll

libcgco.dll: \
    gco_source/LinkedBlockList.o gco_source/graph.o gco_source/maxflow.o \
        gco_source/GCoptimization.o cgco.o
	$(CXX) -shared $(CFLAGS) \
	    gco_source/LinkedBlockList.o \
	    gco_source/graph.o \
	    gco_source/maxflow.o \
	    gco_source/GCoptimization.o \
	    cgco.o \
	    -o libcgco.dll

testMain: \
    gco_source/LinkedBlockList.o gco_source/graph.o gco_source/maxflow.o \
          gco_source/GCoptimization.o testMain.o
	$(CXX) $(CFLAGS)\
        gco_source/LinkedBlockList.o \
        gco_source/graph.o \
        gco_source/maxflow.o \
        gco_source/GCoptimization.o \
        testMain.o \
        -o testMain \

testMain.o: \
	testMain.cpp gco_source/GCoptimization.h
		$(CXX) -c \
    	    -c testMain.cpp \
    	    -o testMain.o

gco.so: \
    gco_source/LinkedBlockList.o gco_source/graph.o gco_source/maxflow.o \
        gco_source/GCoptimization.o
	$(CXX) -shared $(CFLAGS) gco_source/LinkedBlockList.o \
	    gco_source/graph.o \
	    gco_source/maxflow.o \
	    gco_source/GCoptimization.o -o gco.so

gco_source/LinkedBlockList.o: \
    gco_source/LinkedBlockList.cpp \
        gco_source/LinkedBlockList.h
	$(CXX) $(CFLAGS) \
	    -c gco_source/LinkedBlockList.cpp \
	    -o gco_source/LinkedBlockList.o

gco_source/graph.o: \
    gco_source/graph.cpp gco_source/graph.h gco_source/block.h
	$(CXX) $(CFLAGS) \
	    -c gco_source/graph.cpp \
	    -o gco_source/graph.o

gco_source/maxflow.o: \
    gco_source/block.h gco_source/graph.h gco_source/maxflow.cpp
	$(CXX) $(CFLAGS) \
	    -c gco_source/maxflow.cpp \
	    -o gco_source/maxflow.o

gco_source/GCoptimization.o: \
    gco_source/GCoptimization.cpp gco_source/GCoptimization.h \
        gco_source/LinkedBlockList.h gco_source/energy.h gco_source/graph.h \
        gco_source/graph.o gco_source/maxflow.o
	$(CXX) $(CFLAGS) \
	    -c gco_source/GCoptimization.cpp \
	    -o gco_source/GCoptimization.o

cgco.o: \
    cgco.cpp gco_source/GCoptimization.h
	$(CXX) $(CFLAGS) \
	    -c cgco.cpp \
	    -o cgco.o

test_wrapper: \
    test_wrapper.cpp
	$(CXX) -L. test_wrapper.cpp \
	    -o test_wrapper -lcgco

remove_objects:
	$(RM) $(call FixPath, gco_source/*.o)
	$(RM) $(call FixPath, *.o)

clean:
	$(RM) $(call FixPath, *.exe)
