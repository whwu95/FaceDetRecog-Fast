由于不同环境下opencv版本不同，所以在不同机器先编译.so
g++ -shared -fpic -lm -ldl -o libhavon_ffd.so havon_ffd.c
