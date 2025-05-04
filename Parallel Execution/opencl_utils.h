#pragma once

#include <CL/cl.h>
#include <vector>
#include <string>
#define ll long long
using namespace std;

struct OpenCLContext
{
    cl_platform_id platform;
    vector<cl_device_id> devices;
    cl_context context;
    cl_program program;
    cl_command_queue queue;
};

bool setupOpenCL(OpenCLContext &ctx, const string &kernel_file);
void cleanupOpenCL(OpenCLContext &ctx);
bool runRelaxationKernel(OpenCLContext &ctx,
                         vector<ll> &dist,
                         vector<int> &parent,
                         const vector<pair<int, int>> &edges,
                         const vector<ll> &weights);

