#ifndef __SIMPLE_FW_H__
#define __SIMPLE_FW_H__

#include <iostream>
#include <fstream>
#include <string>

#include <CL/cl.h>

#define PROFILING

namespace simple_ocl {
	struct error_t {
		error_t(std::string _s, cl_int _e) : s(_s), e(_e) {}
		void what(){
			std::cout << "ERROR: " << s << " cl_err(" << e << ")\n";
		}
		std::string s;
		cl_int      e;
	};

	inline void check_error(std::string msg, cl_int err){
		if (err != CL_SUCCESS) {
			throw  error_t(msg, err);
		}
	}

	inline void check_resource(bool cond, std::string msg, cl_int err=-1){
		if (!cond){
			throw  error_t(msg, err);
		}
	}

#define LOG_MSG(x) std::cout << x << std::endl;

	template <typename T, size_t N>
	class example_t 
	{
		public:
			typedef bool (*check_result_func_t)(const T* in, const T* out, size_t num);

			example_t(std::string pfname, int dev_type, const char* filename, const char* kernelname, check_result_func_t check_func) :
				m_check_func(check_func),
				m_pf_id(0),
				m_device_id(0),
				m_context(0),
				m_cmd_q(0),
				m_program(0),
				m_kernel(0),
				m_local_size(0),
				m_input(0),
				m_output(0),
				m_event(0) {
					cl_int err;

					// Step-1 Get a Platform

					//Step-1a Try to get 100 platforms on the system....
					cl_platform_id  pfids[100];
					cl_uint         npf;
					err = clGetPlatformIDs(100, pfids, &npf);
					check_error("clGetPlatformIDs failed", err);

					//Step-1b Check for your specific platform
					char cbuff[128];
					for(cl_uint i=0; i < npf; i++){
						err = clGetPlatformInfo(pfids[i],CL_PLATFORM_NAME,128*sizeof(char),cbuff,NULL);
						if(pfname == std::string(cbuff)){
							m_pf_id = pfids[i];
							break;
						}
					}
					check_resource(m_pf_id,std::string("Cannot find platform ") + pfname);
					LOG_MSG("Found platform " << pfname << " id= " << m_pf_id);

					//Step-1c Get a device of the appropriate type
					err = clGetDeviceIDs(m_pf_id, dev_type, 1, &m_device_id, NULL);
					check_error("clGetDeviceIDs failed", err);
					LOG_MSG("Selected device id= " << m_device_id);

					//Step-1d Create a compute context
					m_context = clCreateContext(0, 1, &m_device_id, NULL, NULL, &err);
					check_resource(m_context,"Failed to create a compute context!", err);
					LOG_MSG("Selected context= " << m_context);

					//Step-1e Create a command queue
#ifdef PROFILING
					m_cmd_q = clCreateCommandQueue(m_context, m_device_id, CL_QUEUE_PROFILING_ENABLE, &err);
#else
					m_cmd_q = clCreateCommandQueue(m_context, m_device_id, 0, &err);
#endif
					check_resource(m_cmd_q,"Failed to create a command queue!", err);
					LOG_MSG("Selected command queue = " << m_cmd_q);

					//Step2 Create a program
					//Step2a : Read a Kernel File
					std::ifstream file(filename);
					std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

					const char* source[] = { prog.c_str(), 0 };

					//Step2b : Create the program
					m_program = clCreateProgramWithSource(m_context, 1, source, NULL, &err);
					check_resource(m_program,"Failed to create a program!",err);
					LOG_MSG("Created program = " << m_program);

					//Step2c: Build the program executable
					std::string buildOptions;
					char buf[256]; 
					sprintf(buf,"-D BLK=%d", 512);
					buildOptions += std::string(buf);
					err = clBuildProgram(m_program, 0, NULL, buildOptions.c_str(), NULL, NULL);
					if (err != CL_SUCCESS) {
						size_t len;
						char buffer[2048]; 
						clGetProgramBuildInfo(m_program, m_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
						check_error(std::string("clBuildProgram Failed\n") + buffer, err);
					}

					//Step2d: Create the compute kernel in the program
					m_kernel = clCreateKernel(m_program, kernelname, &err);
					check_resource(m_kernel,std::string("Failed to create compute kernel ") + kernelname, err);
					LOG_MSG("created kernel " << kernelname << ", " << m_kernel);

					//Step3: Get the maximum work group size for executing the kernel on the device
					err = clGetKernelWorkGroupInfo(m_kernel, m_device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(m_local_size), &m_local_size, NULL);
					check_error(std::string("Failed to retrieve kernel work group info ") + kernelname, err);
					LOG_MSG("local workgroup size = " << m_local_size);

					//Step4 : Setup device memory for input/output
					m_input  = clCreateBuffer(m_context, CL_MEM_READ_ONLY,  sizeof(T) * N, NULL, NULL);
					check_resource(m_input,std::string("Failed to create input memory "), -1);
					LOG_MSG("created input " << m_input);

					m_output = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, sizeof(T) * N, NULL, NULL); 
					check_resource(m_output,std::string("Failed to create input memory "), -1);
					LOG_MSG("created output " << m_output);

					//Step5 : setup arguments to the compute kernel
					err = 0;
					err  = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &m_input);
					err |= clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &m_output);
					check_error(std::string("Failed to setup kernel arguments "), err);
					LOG_MSG("Setup kernel arguments correctly");
				}

			void memcopy_device_input(const T* buff){
				cl_int err = clEnqueueWriteBuffer(m_cmd_q, m_input, CL_TRUE, 0, sizeof(T) * N, buff, 0, NULL, NULL);
				check_error(std::string("clEnqueueWriteBuffer"), err);
				LOG_MSG("memcopy_device_input() success");
			}

			void memcopy_device_output(T* buff){
				cl_int err = clEnqueueReadBuffer(m_cmd_q, m_output, CL_TRUE, 0, sizeof(T) * N, buff, 0, NULL, NULL ); 
				check_error(std::string("clEnqueueReadBuffer"), err);
				LOG_MSG("memcopy_device_output() success");
			}

			bool validate_result(const T* in, const T* out, size_t max){
				return m_check_func(in, out, max);
			}

			void launch_kernel(size_t Q){
				//assert Q <= N
				cl_int err;
				size_t global_size = Q;
				err = clEnqueueNDRangeKernel(m_cmd_q, m_kernel, 1, NULL, &global_size, &m_local_size, 0, NULL, &m_event);
				check_error(std::string("launch_kernel() failed!"), err);
				LOG_MSG("NDRange, global= " << global_size << ", local= " << m_local_size);
				err = clWaitForEvents(1,&m_event);
				if (err != CL_SUCCESS){
					throw error_t(std::string("clWaitForEvents() failed!"), err);
				}else{
					std::cout << "clWaitForEvents() event= " << m_event << "\n";
				}
#ifdef PROFILING
				cl_ulong start = 0, end=0;
				err  = clGetEventProfilingInfo(m_event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start,NULL);
				err |= clGetEventProfilingInfo(m_event,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&end,NULL);
				if (err != CL_SUCCESS){
					throw error_t(std::string("clGetEventInfo() start failed!"), err);
				}else{
					std::cout << "clGetEventInfo() event= " << m_event << "\nPROFILING INFO START\n\tstart time= " << start << "\n\tend time= " << end << "\n";
				}
				double duration  = 1.0e-9 * (end - start);
				double bandwidth = sizeof(T) * N * 1.0e-9 / duration;
				std::cout << "\tExec. Time= " << duration << " sec\n";
				std::cout << "\tBandwidth= " << bandwidth << " GBps\n";
				std::cout << "PROFILING INFO END\n";
#endif
			}

			~example_t(){
				clReleaseEvent(m_event);
				clReleaseMemObject(m_output);
				clReleaseMemObject(m_input);
				clReleaseKernel(m_kernel);
				clReleaseProgram(m_program);
				clReleaseCommandQueue(m_cmd_q);
				clReleaseContext(m_context);
			}
		private:
			example_t();
			example_t(const example_t& rhs);
		private:
			check_result_func_t  m_check_func;
			cl_platform_id       m_pf_id;
			cl_device_id         m_device_id;
			cl_context           m_context;
			cl_command_queue     m_cmd_q;
			cl_program           m_program;
			cl_kernel            m_kernel;
			size_t		     m_local_size;
			cl_mem               m_input;
			cl_mem               m_output;
			cl_event             m_event;
	};
}//namespace simple_ocl

#endif
