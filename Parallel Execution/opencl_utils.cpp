#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_utils.h"
#include <fstream>
#include <iostream>
#define ll long long

using namespace std;    

bool setupOpenCL(OpenCLContext &ctx, const string &kernel_file)
{
    cl_int err;

    // Get Platform
    err = clGetPlatformIDs(1, &ctx.platform, NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to find OpenCL platform." << endl;
        return false;
    }

    // Get Devices
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (num_devices == 0)
    {
        cerr << "No GPU devices available. Falling back to CPU." << endl;
        err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);
    }
    ctx.devices.resize(num_devices);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_ALL, num_devices, ctx.devices.data(), NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to get OpenCL devices." << endl;
        return false;
    }

    // Create Context
    ctx.context = clCreateContext(NULL, num_devices, ctx.devices.data(), NULL, NULL, &err);
    if (!ctx.context)
    {
        cerr << "Failed to create OpenCL context." << endl;
        return false;
    }

    // Read kernel source
    ifstream file(kernel_file);
    if (!file.is_open())
    {
        cerr << "Cannot open kernel file: " << kernel_file << endl;
        return false;
    }
    string source((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    const char *source_str = source.c_str();

    // Create program
    ctx.program = clCreateProgramWithSource(ctx.context, 1, &source_str, NULL, &err);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to create OpenCL program." << endl;
        return false;
    }

    // Build Program
    err = clBuildProgram(ctx.program, num_devices, ctx.devices.data(), NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Error building OpenCL program." << endl;
        size_t len = 0;
        clGetProgramBuildInfo(ctx.program, ctx.devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        vector<char> build_log(len);
        clGetProgramBuildInfo(ctx.program, ctx.devices[0], CL_PROGRAM_BUILD_LOG, len, build_log.data(), NULL);
        cerr << "Build Log:\n"
                  << build_log.data() << endl;
        return false;
    }

    // Create queue for device 0
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.devices[0], 0, &err);
    if (!ctx.queue)
    {
        cerr << "Failed to create command queue." << endl;
        return false;
    }

    return true;
}

void cleanupOpenCL(OpenCLContext &ctx)
{
    if (ctx.queue)
        clReleaseCommandQueue(ctx.queue);
    if (ctx.program)
        clReleaseProgram(ctx.program);
    if (ctx.context)
        clReleaseContext(ctx.context);
}

bool runRelaxationKernel(OpenCLContext &ctx,
                         vector<ll> &dist,
                         vector<int> &parent,
                         const vector<pair<int, int>> &edges,
                         const vector<ll> &weights)
{
    cl_int err;
    size_t num_edges = edges.size();
    bool success = true;

    // Declare variables at the top to avoid goto issues
    size_t global_size = num_edges;
    size_t local_size = 64; // Adjust based on your device capabilities

    // Check for empty input
    if (num_edges == 0 || dist.empty() || parent.empty())
    {
        cerr << "Error: Empty input data for OpenCL kernel" << endl;
        return false;
    }

    // Create buffers
    cl_mem dist_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(ll) * dist.size(), dist.data(), &err);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to create dist buffer: " << err << endl;
        return false;
    }

    cl_mem parent_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(int) * parent.size(), parent.data(), &err);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to create parent buffer: " << err << endl;
        clReleaseMemObject(dist_buf);
        return false;
    }

    // Create a flattened array of edge indices for OpenCL
    vector<cl_int2> cl_edges(num_edges);
    for (size_t i = 0; i < num_edges; i++)
    {
        cl_edges[i].s[0] = edges[i].first;
        cl_edges[i].s[1] = edges[i].second;
    }

    cl_mem edges_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_int2) * num_edges, const_cast<void *>(static_cast<const void *>(edges.data())), &err);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to create edges buffer: " << err << endl;
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        return false;
    }

    cl_mem weights_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(ll) * num_edges, const_cast<void *>(static_cast<const void *>(weights.data())), &err);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to create weights buffer: " << err << endl;
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        return false;
    }

    cl_kernel kernel = clCreateKernel(ctx.program, "relax_edges", &err);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to create kernel: " << err << endl;
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        clReleaseMemObject(weights_buf);
        return false;
    }

    // Set kernel args
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dist_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &parent_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &edges_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &num_edges);

    if (err != CL_SUCCESS)
    {
        cerr << "Failed to set kernel arguments: " << err << endl;
        clReleaseKernel(kernel);
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        clReleaseMemObject(weights_buf);
        return false;
    }

    // Ensure global_size is multiple of local_size
    global_size = ((global_size + local_size - 1) / local_size) * local_size;

    err = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to launch kernel: " << err << endl;
        clReleaseKernel(kernel);
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        clReleaseMemObject(weights_buf);
        return false;
    }

    // Wait for kernel to complete
    err = clFinish(ctx.queue);
    if (err != CL_SUCCESS)
    {
        cerr << "Error waiting for kernel completion: " << err << endl;
        clReleaseKernel(kernel);
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        clReleaseMemObject(weights_buf);
        return false;
    }

    // Read back results
    err = clEnqueueReadBuffer(ctx.queue, dist_buf, CL_TRUE, 0, sizeof(ll) * dist.size(), dist.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to read dist buffer: " << err << endl;
        clReleaseKernel(kernel);
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        clReleaseMemObject(weights_buf);
        return false;
    }

    err = clEnqueueReadBuffer(ctx.queue, parent_buf, CL_TRUE, 0, sizeof(int) * parent.size(), parent.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Failed to read parent buffer: " << err << endl;
        clReleaseKernel(kernel);
        clReleaseMemObject(dist_buf);
        clReleaseMemObject(parent_buf);
        clReleaseMemObject(edges_buf);
        clReleaseMemObject(weights_buf);
        return false;
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseMemObject(dist_buf);
    clReleaseMemObject(parent_buf);
    clReleaseMemObject(edges_buf);
    clReleaseMemObject(weights_buf);

    return success;
}