include ./arch.gnu
# OPTIMIZATION = -fast
# OPTIMIZATION = -O3
# DEBUG += -g

app:		cardiacsim cardiacsim-serial 

OBJECTS1 = cardiacsim.o splot.o cmdLine.o

OBJECTS2 = cardiacsim-serial.o splot.o cmdLine.o

cardiacsim:		$(OBJECTS1)
	$(C++LINK) $(LDFLAGS) -o $@ $(OBJECTS1)  $(LDLIBS)

cardiacsim-serial:		$(OBJECTS2)
	$(C++LINK) $(LDFLAGS) -o $@ $(OBJECTS2)  $(LDLIBS)


clean:	
	$(RM) *.o cardiacsim *~;
	$(RM) core;
