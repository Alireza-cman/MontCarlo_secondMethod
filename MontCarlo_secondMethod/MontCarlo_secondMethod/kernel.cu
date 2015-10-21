
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "curand_kernel.h"
#include <ctime>
#include <fstream>

#define NE WA*HA //Total number of random numbers 
#define WA 2   // Matrix A width
#define HA 2   // Matrix A height
#define SAMPLE 100 //Sample number
#define BLOCK_SIZE 2 //Block size
#define imin(a,b) (a<b?a:b)


using namespace std;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		// if (abort) exit(code);
	}
}


long int  N = 1 << 17;
int num_T = 200;
const int threadsPerBlock = 256;
const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
const int blocksPerGridForRand = (N*num_T + threadsPerBlock - 1) / threadsPerBlock;
__global__ void cudaRand(double *d_out, const long int havij)
{
	long int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock() + i, 0, 0, &state); //
	while (i < havij)
	{
		d_out[i] = curand_normal_double(&state);
		i += blockDim.x * gridDim.x;
	}


}
__global__ void blackSchole(double s, double mu, double V, double V_pow, double T, double sqrt_T, double *randnum, double *output, const int N)
{
	long int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//__shared__ double cache[threadsPerBlock];
	double temp = s;

	//long int cacheIndex = threadIdx.x;
	while (tid < N)
	{
		//temp = temp*exp((mu-0.5*V_pow)*T + V*sqrt(T)*randnum[tid]);
		for (int i = 0; i < 200; i++)
		{
			temp = temp*exp((mu - 0.5*V)*T + V*sqrt_T*randnum[tid * 200 + i]);
			output[tid * 200 + i] = temp;

		}
		tid += blockDim.x * gridDim.x;
		temp = s;
	}


}

int main()
{
	cudaFree(0);
	ofstream myfile, randome;
	myfile.open("AlirzaLogFile.txt");
	randome.open("randome.txt");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//cout<<blocksPerGridForRand<<endl<<blocksPerGrid;
	long int havij = num_T*N;
	//double *h_v ;
	//double *h_v = (double*)malloc(havij+2);
	double *h_v = new double[N*num_T];
	//cudaHostAlloc((void**)&h_v,havij*sizeof(double),1);
	double S = 1, mu = 0.05, V = 0.02, T = 10;
	T = T / num_T;
	double V_pow = V*V;
	double sqrt_T = sqrt(T);
	double *d_out;
	double *output;
	cudaMalloc((void**)&d_out, havij * sizeof(double));
	cudaMalloc((void**)&output, havij * sizeof(double));
	cudaEventRecord(start, 0);
	cudaRand << < blocksPerGrid, threadsPerBlock >> > (d_out, havij);
	/*gpuErrchk(cudaMemcpy(h_v, d_out,  num_T * N * sizeof(double), cudaMemcpyDeviceToHost));
	for (long int i = 0; i < N; i++)
	{
	for (int j = 0 ; j < num_T ; j++)
	{
	randome <<h_v[i*num_T + j]<<"\t";
	}
	randome<<endl;
	}*/
	cudaDeviceSynchronize();
	blackSchole << < blocksPerGrid, threadsPerBlock >> >(S, mu, V, V_pow, T, sqrt_T, d_out, output, N);
	cudaDeviceSynchronize();
	cudaMemcpy(h_v, output, num_T * N * sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %3.1f ms\n", elapsedTime);
	for (long int i = 0; i < N; i++)
	{
		for (int j = 0; j < num_T; j++)
		{
			myfile << h_v[i*num_T + j] << "\t";
		}
		myfile << endl;
	}

	getchar();
	myfile.close();
	randome.close();
	cudaFree(d_out);
	cudaFree(output);
	delete[] h_v;
	//cudaFreeHost(h_v);
	return 0;


}

