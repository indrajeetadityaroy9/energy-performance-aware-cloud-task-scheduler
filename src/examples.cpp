#include "mcc.hpp"

using namespace std;

vector<vector<Task>> example_graphs() {
    static const map<int, vector<double>> core_execution_times = {
        {1, {9, 7, 5}}, {2, {8, 6, 5}}, {3, {6, 5, 4}}, {4, {7, 5, 3}},
        {5, {5, 4, 2}}, {6, {7, 6, 4}}, {7, {8, 5, 3}}, {8, {6, 4, 2}},
        {9, {5, 3, 2}}, {10,{7,4,2}},  {11,{10,7,4}}, {12,{11,8,5}},
        {13,{9,6,3}}, {14,{12,8,4}}, {15,{10,7,3}}, {16,{11,7,4}},
        {17,{9,6,3}}, {18,{12,8,5}}, {19,{10,7,4}}, {20,{11,8,5}}
    };

    static const array<double,3> cloud_execution_times = {3, 1, 1};

    vector<int> ten_task_graph_task_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<int> twenty_task_graph_task_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    // Graph 1
    vector<pair<int,int>> graph1_edges = {
        {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
        {2,8}, {2,9},
        {3,7},
        {4,8}, {4,9},
        {5,9},
        {6,8},
        {7,10},
        {8,10},
        {9,10}
    };

    //Graph 2
    vector<pair<int,int>> graph2_edges = {
        {1,2}, {1,3},
        {2,4}, {2,5},
        {3,5}, {3,6},
        {4,6},
        {5,7},
        {6,7}, {6,8},
        {7,8}, {7,9},
        {8,10},
        {9,10}
    };

    //Graph 3
    vector<pair<int,int>> graph3_edges = {
        {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
        {2,7}, {2,8},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18}, {14,19},
        {15,19},
        {16,19},
        {17,20},
        {18,20},
        {19,20}
    };

    //Graph 4
    vector<pair<int,int>> graph4_edges = {
        {1,7},
        {2,7},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18}, {14,19},
        {15,19},
        {16,19},
        {17,20},
        {18,20},
        {19,20}
    };

    //Graph 5
    vector<pair<int,int>> graph5_edges = {
        {1,4}, {1,5}, {1,6},
        {2,7}, {2,8},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18},
        {15,19},
        {16,19},
        {18,20},
    };

    vector<vector<Task>> all_graphs;
    all_graphs.push_back(create_task_graph(ten_task_graph_task_ids, core_execution_times, cloud_execution_times, graph1_edges));
    all_graphs.push_back(create_task_graph(ten_task_graph_task_ids, core_execution_times, cloud_execution_times, graph2_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph3_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph4_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph5_edges));

    return all_graphs;
}
