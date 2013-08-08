CXX=        g++
BOOST_BASE= C:/MinGW/
CFLAGS=     -m32 -Wall -g

#OCLBASE= "C:\Program Files (x86)\Intel\OpenCL SDK\2.0"
#IFLAGS = -I $(OCLBASE)/include
#LFLAGS= -L $(OCLBASE)/lib/x64
#LFLAGS = -L $(OCLBASE)/lib/x86

OCLBASE= "C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.2\OpenCL\common"
IFLAGS=     -I $(OCLBASE)/inc
#LFLAGS=     -L $(OCLBASE)/lib/x64
LFLAGS=     -L $(OCLBASE)/lib/Win32

#OCLBASE= "C:\Program Files (x86)\AMD APP"
#IFLAGS=     -I $(OCLBASE)/include
#LFLAGS=     -L $(OCLBASE)/lib/x86
#LFLAGS=     -L $(OCLBASE)/lib/x86_64

LIBS=       -lOpenCL

fold.exe : fold.cpp
	$(CXX) $(CFLAGS) $(IFLAGS) $< -o $@ $(LFLAGS) $(LIBS)
