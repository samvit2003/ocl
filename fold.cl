__kernel void square(__global float* input, __global float* output) {
   int i = get_global_id(0);
   output[i] = input[i] * input[i];
}

__kernel void fold_0(__global int* input, __global int* output) 
{
   int i = get_global_id(0);
   int N = get_global_size(0);

   if(i < N && i % 2 == 0){
       output[i] = input[i] + input[i+1];
   }
}

__kernel void fold_1(__global int* input, __global int* output) 
{
   int i = get_global_id(0);
   int N = get_global_size(0);

   int p = get_local_id(0);
   int g = get_group_id(0);

   __local int   buff[BLK];

   buff[p] =  input[i];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int s=BLK/2; s >= 1; s = s/2){

      if(p < s){
          buff[p] += buff[p+s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

   }
   
   if(p == 0){
   	//output[i/512] = buff[p];
   	output[g] = buff[p];
   }
}



__kernel void fold_2(__global int* input, __global int* output) 
{
   int i = get_global_id(0);
   int N = get_global_size(0);

   int p = get_local_id(0);
   int g = get_group_id(0);

   __local int   buff[BLK];

   buff[p] =  input[i] + input[i+N];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int s=BLK/2; s >= 1; s = s/2){

      if(p < s){
          buff[p] += buff[p+s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

   }
   
   if(p == 0){
   	//output[i/512] = buff[p];
   	output[g] = buff[p];
   }
}

__kernel void fold_3(__global int* input, __global int* output) 
{
   int i = get_global_id(0);
   int N = get_global_size(0);

   int p = get_local_id(0);
   int g = get_group_id(0);

   __local int   buff[BLK];

   buff[p] =  input[i] + input[i+N];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int s=BLK/2; s > 16 ; s = s/2){

      if(p < s){
          buff[p] += buff[p+s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(p < 16){
          buff[p] += buff[p+16];
          buff[p] += buff[p+8];
          buff[p] += buff[p+4];
          buff[p] += buff[p+2];
          buff[p] += buff[p+1];
   }
   
   if(p == 0){
   	output[g] = buff[p];
   }
}

