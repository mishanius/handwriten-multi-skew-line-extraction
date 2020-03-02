//
// Created by michael on 2/10/20.
//
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include "gco_source/GCoptimization.h"
#include "testMain.h"
#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include<iostream>

using namespace std;


int populate_matrix_test(int * matrix, int num_labels,  const std::string& input){
    string temp;
    std::ifstream file(input);
    std::string                cell;
    int number, index_row, index_col;
    index_row=0;
    while (file >> temp) {
        std::stringstream          lineStream(temp);
        index_col=0;
		if(num_labels==0){
			number = stoi(temp);
			matrix[index_row]=number;
		}
		else{
			while(std::getline(lineStream,cell, ','))
			{
				number = stoi(cell);
				matrix[index_row*num_labels+index_col]=number;
				index_col++;
			}
		}
        index_row++;
    }
    return 0;
}

int main(int argc, char **argv)
{
    std::string data_costCsv = "data_cost.csv";
    std::string labelsCsv = "label_cost.csv";
    std::string neighborsCsv = "neighbors.csv";
    std::string smoothCsv = "smooth_cost.csv";
	int num_of_sites = stoi(argv[1]);
    int num_of_labels = stoi(argv[2]);
    int num_of_neigboors_pairs = stoi(argv[3]);
	if(argc>4){
		data_costCsv = argv[4];
		labelsCsv = argv[5];
		neighborsCsv = argv[6];
		smoothCsv = argv[7];
	}
    GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_of_sites,num_of_labels);
    int *data_matrix = new int[num_of_sites*num_of_labels];
    int *labels = new int[num_of_labels];
    int *neighbors = new int[num_of_neigboors_pairs*3];
    int *smooth = new int[num_of_labels*num_of_labels];
    populate_matrix_test(data_matrix, num_of_labels, data_costCsv);
	populate_matrix_test(labels, 0, labelsCsv);
	populate_matrix_test(neighbors, 3, neighborsCsv);
	populate_matrix_test(smooth, num_of_labels, smoothCsv);
	gc->setDataCost(data_matrix);
    gc->setSmoothCost(smooth);
	gc->setLabelCost(labels);
    for (int y = 0; y < num_of_neigboors_pairs; y++ ){
        gc->setNeighbors(neighbors[y*3],neighbors[y*3+1],neighbors[y*3+2]);
    }
    gc->expansion();
    for ( int  i = 0; i < num_of_sites; i++ )
        cout << gc->whatLabel(i) << ",";
    return 0;
}
