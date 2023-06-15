/**
 * @Author: Sadamori Kojaku
 * @Date:   2022-11-02 13:02:40
 * @Last Modified by:   Sadamori Kojaku
 * @Last Modified time: 2022-11-28 14:59:59
 */
/*
 *   sbm version 1.1, release date 30/05/2012
 *   Copyright 2012 Aurelien Decelle, Florent Krzakala, Lenka Zdeborova and Pan Zhang
 *   sbm is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.

 *   sbm is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
*/
//{{{ defines and header files
#ifndef __BM__
#define __BM__
#define Q_PERMU 8
#define EPS 1.0e-50
#define myexp exp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <ctime>
#include <map>
#include <algorithm>
#include <assert.h>

#define PI 3.14159265358
#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)
//#define OVL_NORM
const char deli[1024]="\t ";
//}}}
using namespace std;
class blockmodel
{
public:
	//{{{ variables
	vector < vector<int> > perms;//permutations
	int vflag;
	int LARGE_DEGREE;

	//graph part
	int N, Q, Q_true, M, graph_max_degree;
	// N: number of nodes.  Q: number of colors of the model!!!   Q_true: number of colors given
	// M: number of edges	graph_max_degree: maximum degree of the graph.
	vector <string> graph_ids;//it stores id (in string) of nodes.
	vector < vector <int> > graph_neis, graph_neis_inv, graph_edges;
	// graph_neis[i][j] is the number of the j^th neighbor of i
	// graph_neis_inv[i][j] is the neighboring number of spin graph_neis[i][j] that correspond to i
	// graph_edges[] stores all (undirected!) edges of the graph.
	vector < vector<int> > groups_true,groups_infer;
	vector < int > conf_true,conf_infer;// conf_true stores configuration(assignment, or colors or communities) that given, and conf_infer stores configuration that infered.
	vector <double> graph_di,bm_conductance;//degree of nodes, normalized by maximum degree.
	double graph_min_conductance;

	//block model
	vector < double > na, nna, na_true, na_expect, nna_expect, eta, logeta, argmax_marginals; //nna is normalized na for degree corrected model.
	vector < vector < double > > pab, cab, logcab, cab_true, pab_true, cab_expect;
	int bm_dc;

	//learning and inference
	vector <double> h;
	int learning_max_time;
	int bp_last_conv_time;
	double bp_last_diff;
	vector <double> field_iter, normtot_psi, pom_psi, exph, maxpom_psii_iter;
	vector < vector < vector <double> >  > psi;  // psi[i][j] is the message from j-> i
	vector < vector <double> > psii, real_psi, psii_iter; // real_psi is the total marginal of spin i

	//random number generator
	int seed;
	unsigned myrand, ira[256];
	unsigned char ip, ip1, ip2, ip3;
	//}}}
	//{{{ functions
	//permutation
	void init_perms();
	//random number generator
	unsigned init_rand4init(void);
	void init_random_number_generator(int);
	//graph
	void graph_build_neis_inv();
	void graph_read_graph(string);
	void graph_write_gml(const char *);
	void graph_write_csv(const char *, double);
	void graph_write_spm(const char *);
	void graph_write_A(const char *);
	void graph_gen_graph(int, vector<double> , vector<double >);
	//block model
	blockmodel();
	void set_Q(int);
	void set_vflag(int);
	void set_dc(int);
	void bm_allocate();
	void bm_show_na_cab();
	void bm_show_na_cab(string fname);
	void bm_init_uniform();
	void bm_init_random();
	void bm_rescale_cab();
	void bm_init_random_assort();
	void bm_init(vector<double>, vector<double>);
	void bm_build_na_cab_from_conf();
	//message passing
	void bp_init_h();
	void bp_init_m(double);
	void bp_init(int);
	void bp_allocate();
	double bp_compute_f();
	double compute_log_likelihood(double);
	double bp_compute_cab_expect();
	double bp_iter_update_psi(int, double);
	double bp_iter_update_psi_large_degree(int, double);
	int bp_converge(double, int, int, double);

	double nmf_compute_f();
	double nmf_compute_cab_expect();
	int nmf_converge(double, int, int,double);
	void nmf_init_h();
	void nmf_init_m(double);
	void nmf_init(int);
	void nmf_allocate();
	double nmf_iterate(int, double);

	void compute_argmax_conf();
	double compute_na_expect();
	double compute_overlap();
	double compute_overlap_marg();
	void compute_conductance();
	double compute_overlap_fraction();
	double compute_config_ovl();
	double compute_overlap_EA();
	double compute_na_expect_degree();
	double compute_argmax_energy();

	//Output part
	void show_marginals(int );
	void output_marginals(string );
	void output_f_ovl(int);


	//Learning and inference
	void EM_learning(int, int, double, int, double, double, int, double, bool);
	void walk_learning(int mode, int init_flag, double conv_crit, int time_conv, double dumping_rate, int learning_max_time,int blocktime ,double wstep, bool no_learn_pa);
	void do_inference(int, int, double, int, double);
	void EM_step(double,bool);
	void shuffle_seq(vector <int> &sequence);
	void spec(string fname, string origin_fname,int mode);
};
//}}}
string get_std_from_cmd(const char*);
vector <string> strsplit(const string& , const string&);
#endif

