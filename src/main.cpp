#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include<load_save.h>
#include<blur_ops.h>
#include<edge_detection.h>

using namespace std;

size_t numRows, numCols;

uchar4* load_image_in_GPU(string filename) {
	// Load the image into main memory
	uchar4 *h_image = NULL, *d_in = NULL;
	loadImageRGBA(filename, &h_image, &numRows, &numCols);
 	// Allocate memory to the GPU
	cudaMalloc((void **) &d_in, numRows * numCols * sizeof(uchar4));
	cudaMemcpy(d_in, h_image, numRows * numCols * sizeof(uchar4), cudaMemcpyHostToDevice);
	// No need to keep this image in RAM now.
	delete h_image;
	return d_in;
}

int hex_to_int(string hexStr) {
	int i;
	stringstream ss;
    ss << std::hex << hexStr;
	ss >> i;
	return i;
}

uchar4 hex_to_uchar4_color(string& color) {
	int r = hex_to_int(color.substr(0, 2));
	int g = hex_to_int(color.substr(2, 2));
	int b = hex_to_int(color.substr(4, 2));
	return make_uchar4(r, g, b, 255);
}

int main(int argc, char **argv) {

	string input_file = "original_100.jpg";
	string output_file = "d_gauss_100.jpg";
	uchar4 *d_in = load_image_in_GPU(input_file);
	uchar4 *h_out = NULL;

	// Performing the required operation
	int amount = 10;
	if(amount % 2 == 0)
		amount++;
	h_out = blur_ops(d_in, numRows, numCols, amount);

	cudaFree(d_in);
	if(h_out != NULL)
		saveImageRGBA(h_out, output_file, numRows, numCols);

	string str = "convert "; 
    str = str + "original_100.jpg " + "original_100.pgm";

    const char *command = str.c_str();
    system(command);

	str = "convert "; 
    str = str + "d_gauss_100.jpg " + "d_gauss_100.pgm";

    command = str.c_str();
    system(command);

    char *t1 = "original_100.pgm";
    char *t2 = "d_gauss_100.pgm";
    char *t3 = "h_original_100_edge_50.pgm";
    char *t4 = "d_original_100_edge_50.pgm";
    char *t5 = "h_gauss_100_edge_50.pgm";
    char *t6 = "d_gauss_100_edge_50.pgm";

	edgeDetection(t1, t3, t4);
	edgeDetection(t2, t5, t6);

}
