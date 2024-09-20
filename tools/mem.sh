nvcc mem.cu
nvprof --metrics gld_transactions,gst_transactions ./a.out
