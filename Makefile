CC=clang
CXX=clang++

CPPFLAGS=-g -I./thirdparty/libtorch/include/ -std=c++14
LDFLAGS=-pthread
OBJS=main.o

.PHONY: save-package
save-package:
	pip freeze > requirements.txt

.PHONY: install-package
install-package:
	pip install -r requirements.txt

$(OBJS): Makefile

main: $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -f *.o main
