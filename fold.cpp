#include "framework.hpp"

bool check_fold(const int* in, const int* out, size_t max) {
	int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
	for(size_t i=0; i<max; i++){
		if(i < max/4){ sum0 += in[i];
		}else if(i < 2*max/4){ sum1 += in[i];
		}else if(i < 3*max/4){ sum2 += in[i];
		}else if(i < 4*max/4){ sum3 += in[i];
		}
	}

	std::cout << "check_fold() sum0= " << sum0  << ", sum1= " << sum1  << ", sum2= " << sum2  << ", sum3= "  << sum3 << "\n";
	std::cout << "check_fold() out[0]= " << out[0]  << ", out[1]= " << out[1]  << ", out[2]= " << out[2]  << ", out[3]= "  << out[3] << "\n";

	return (out[0] == sum0 && out[1] == sum1 && out[2] == sum2 && out[3] == sum3);
}


int main(int argc, char* argv[])
{
	try {
		constexpr size_t data_size = 1 << 11;

		simple_ocl::example_t<int,data_size> ex1("NVIDIA CUDA", CL_DEVICE_TYPE_GPU, "fold.cl", "fold_1", check_fold);
		//simple_ocl_example_t<float,data_size> ex1("Intel(R) OpenCL", CL_DEVICE_TYPE_CPU, "jnk.cl", "square", check_square);

		int* data   = new int[data_size];
		int* result = new int[data_size];

		//initialize input
		//std::random_device              rd;
		//std::mt19937                    gen(rd());
		//std::uniform_int_distribution<> dis(1,6);
		int sum0 = 0;
		int sum1 = 0;
		int sum2 = 0;
		int sum3 = 0;
		for(size_t i=0; i<data_size; i++){
			if(i < 512){
				data[i] = i+1;
				sum0 += data[i];
		        	//std::cout << "in[" << i << "] = " << data[i] << "\n";
			}else if(i < 1024){
				data[i] = i+1;
				sum1 += data[i];
		        	//std::cout << "in[" << i << "] = " << data[i] << "\n";
			}else if(i < 512+1024){
				data[i] = -(i+1);
				sum2 += data[i];
		        	//std::cout << "in[" << i << "] = " << data[i] << "\n";
			}else{
				data[i] = -(i+1-512);
				sum3 += data[i];
		        	//std::cout << "in[" << i << "] = " << data[i] << "\n";
			}
		}
	        std::cout << "data() sum0= " << sum0  << ", sum1= " << sum1  << ", sum2= " << sum2  << ", sum3= "  << sum3 << "\n";

		ex1.memcopy_device_input(data);
		ex1.launch_kernel(data_size);
		ex1.memcopy_device_output(result);

		// Validate our results
		bool val = ex1.validate_result(data,result,data_size);
		std::string x = val ? "TEST PASSED" : "TEST FAILED";
		std::cout << x << std::endl;

		delete [] data;
		delete [] result;
	}catch(simple_ocl::error_t& e){
                e.what();
		return 1;
        }catch(...){
                return 1;
        }
  
  return 0;
}

