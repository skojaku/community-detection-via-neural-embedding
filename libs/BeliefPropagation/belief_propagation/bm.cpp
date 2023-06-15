/**
 * @Author: Sadamori Kojaku
 * @Date:   2022-11-02 13:02:40
 * @Last Modified by:   Sadamori Kojaku
 * @Last Modified time: 2022-11-28 14:59:19
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
#include "bm.h"
#include <sstream>
#include <cstring>
#include <string>
//{{{void blockmodel::shuffle_seq(vector <int> &sequence)
void blockmodel::shuffle_seq(vector <int> &sequence)
{
	//random shuffle the sequence
	//I put this function inside class blockmodel because I need to use the random number generator inside the class.
	for(int i=0;i<N;i++) sequence[i]=i;
	int tmp;
	for(int i=0;i<N;i++){
		int tmpindex=int(FRANDOM*N);
		//int tmpindex=int(rg->rdflt()*(N-i));
		tmp=sequence[tmpindex];
		sequence[tmpindex]=sequence[N-i-1];
		sequence[N-i-1]=tmp;
	}
}
//}}}
//set parameters
blockmodel::blockmodel() { bm_dc=0; }
void blockmodel::set_Q(int Q_) { Q=Q_; init_perms(); }
void blockmodel::set_vflag(int vflag_) { vflag=vflag_; }
void blockmodel::set_dc(int bm_dc_) { bm_dc=bm_dc_; } //set degree corrected model bm_dc_= 0,1,2
//graph part
//{{{ void blockmodel:: graph_read_graph(string fname)
void blockmodel:: graph_read_graph(string fname)
{
	//read from file "fname" to build graph_ids[] graph_neis[][] graph_edges[] graph_neis_inv[][] groups_true[][]
	//and set N M and graph_max_degree
	string snode=string("node"),sbegin=string("["), send=string("]"), sedge=string("edge"),sid=string("id"),svalue=string("value"),ssource=string("source"),starget=string("target");
	map <string,int> id2idx;
	map <string, string> id2value;
	string tmpstr;
	ifstream fin(fname.c_str());
	graph_ids.clear();
	groups_true.clear();
	map <string, int> value2color;
	cout<<"reading "<<fname<<"... "<<flush;
	assert(fin.good()&&"I can not open the file that contains the graph.");
	int idx=0,color_idx=0;
	while(fin >> tmpstr){ // read something if it is possible to read all nodes and build node structures.
		if(tmpstr == snode){//node section begins, let's read id and value.
			fin >> tmpstr;
			assert(tmpstr == sbegin && "[ should follow node");
			bool endnode=false;
			string myid("not_set");
			for(int readtimes=0;readtimes<100;readtimes++){//read at most 100 times to scan all parameters of this node
				fin >> tmpstr;
				if(tmpstr == send) {
					endnode=true;
					break;
				}else if(tmpstr == sid){//read an id and assign an index to this id
					fin >> myid;
					assert(id2idx.count(myid) == 0&&"multi-definition of node in graph file!");
					id2idx[myid] = idx++;
					graph_ids.push_back(myid);
				}else if(tmpstr == svalue){//read a value to an id
					assert(myid != string("not_set") && "id should be given before value");
					fin >> tmpstr;
					id2value[myid] = tmpstr;
					if(value2color.count(tmpstr) == 0){
						vector <int> tmpvec; tmpvec.clear();
						groups_true.push_back(tmpvec);
						value2color[tmpstr] = color_idx++;
					}
				}else{
					//cout<<"this is "<<tmpstr<<" which should be ignored"<<endl;
				}
			}
			assert(endnode && "there must be somthing wrong because I can not find ]");
		}
	}
	fin.close();
	N=graph_ids.size();

	graph_neis.resize(N);
	graph_di.resize(N);
	for(int i=0;i<N;i++) graph_neis[i].resize(0);
	graph_edges.resize(0);
	fin.open(fname.c_str());
	int i=-1,j=-1;
	while(fin >> tmpstr){ // read again the file for edges and to build edge structrues.
		if(tmpstr == sedge){//edge section begins, let's read pairs of nodes
			fin >> tmpstr;
			assert(tmpstr == sbegin && "[ should follow edge");
			fin >> tmpstr;
			while(tmpstr != ssource) fin>>tmpstr;
//			assert(tmpstr == ssource && "there should be souce following edge [");
			fin >> tmpstr;
			i=id2idx[tmpstr];
			fin >> tmpstr;
			assert(tmpstr == starget && "there should be target following source");
			fin >> tmpstr;
			j=id2idx[tmpstr];
			assert(i<N && i>=0 && j<N && j>=0 && "head of this edge does not exist in the node database!");
			bool duplicate=false;
			for(int idxij=0;idxij<graph_neis[i].size();idxij++){
				if(graph_neis[i][idxij] == j){
					duplicate = true;
					break;
				}
			}
			if( !duplicate ){
				for(int idxji=0;idxji<graph_neis[j].size();idxji++){
					if(graph_neis[j][idxji] == i){
						duplicate = true;
						break;
					}
				}
			}
			//assert(!duplicate && "there are duplicated edges! Make sure the file is OK");
			if( !duplicate){
				graph_neis[i].push_back(j);
				graph_neis[j].push_back(i);

				vector <int> tmpvec; tmpvec.clear();
				tmpvec.push_back(i);
				tmpvec.push_back(j);
				graph_edges.push_back(tmpvec);
			}
		}
	}
	fin.close();
	M=graph_edges.size();
	Q_true=groups_true.size();

	cout<<graph_ids.size()<<" nodes "<<groups_true.size()<<" groups, "<<graph_edges.size()<<" edges, c="<<2.0*M/N<<flush;

	graph_build_neis_inv();//build neighbor matrix graph_neis[][] and graph_neis_inv[][] from edge matrix graph_edges.
	conf_true.resize(N);//true assignments
	for(map<string,string>::iterator it=id2value.begin();it != id2value.end(); ++it){
		int idx=id2idx[it->first];
		int color=value2color[it->second];
		conf_true[idx] = color;
		groups_true[color].push_back(idx);
	}
	graph_max_degree=-50;
	for(int i=0; i<N; i ++){
		if(int(graph_neis[i].size()) >= graph_max_degree) graph_max_degree=graph_neis[i].size();
	}
	if(bm_dc==1) {
		for(int i=0;i<N;i++) graph_di[i]=1.0*graph_neis[i].size()/graph_max_degree;
	}else if(bm_dc==2) {
		for(int i=0;i<N;i++) graph_di[i]=1.0*graph_neis[i].size();
	}else {
		for(int i=0;i<N;i++) graph_di[i]=1.0*graph_neis[i].size();
	}
	//for(int i=0;i<N;i++) graph_di[i]=1.0;
	//for(int i=0;i<N;i++) cout<<graph_neis[i].size()<<" "<<flush; cout<<endl;
	cout<<" max_degree="<<graph_max_degree<<endl;
	//for(unsigned int i=0;i<graph_ids.size();i++) cout<<i<<" "<<graph_ids[i]<<endl;
	//for(unsigned int q=0;q<groups_true.size();q++) cout<<"group "<<q<<" size: "<<groups_true[q].size()<<endl;
}
//}}}
//{{{ void blockmodel:: graph_build_neis_inv()
void blockmodel:: graph_build_neis_inv()
{
	//build graph_neis_inv
	graph_neis_inv.resize(N);
	for (int i=0;i<N;i++){
		graph_neis_inv[i].resize(graph_neis[i].size());
		for (int idxij=0;idxij<graph_neis[i].size();idxij++){
			int j=graph_neis[i][idxij];
			int target=0;
			for (int idxji=0;idxji<graph_neis[j].size();idxji++){
				if (graph_neis[j][idxji]==i){
					target = idxji;
					break;
				}
			}
			graph_neis_inv[i][idxij]=target;
		}
	}
}
//}}}
//{{{void blockmodel:: graph_gen_graph(int N_, int seed_, int Q_, vector<double> pa_, vector< vector<double >cab_)
void blockmodel:: graph_gen_graph(int N_,vector<double> pa_, vector<double > cab_)
{
	//generate graph to build graph_ids[] graph_neis[][] graph_edges[] graph_neis_inv[][] groups_true[][]
	//and set N M and graph_max_degree
	N=N_;
	Q_true=Q;
	groups_true.resize(Q);
	for (int q=0;q<Q;q++) groups_true[q].resize(0);
	graph_neis.resize(N);
	conf_true.resize(N);

	bm_allocate();
	bm_init(pa_, cab_);//init na, eta, pa and cab
//	cout<<"True matrices:"<<endl;
//	bm_show_na_cab();
//	cout<<"Generating graph... "<<flush;
	// init communitites
	vector<int> seq;
	for(int i=0;i<N;i++) seq.push_back(i);
//	shuffle_seq(seq);
	int num=0;
	for (int q=0;q<Q;q++){
		for (int j=0;j<na[q];j++) conf_true[seq[num++]]=q;
	}
	for (int i=0;i<N;i++) groups_true[conf_true[i]].push_back(i);
	graph_edges.clear();
	for (int q1=0;q1<Q;q1++){
		for (int q2=q1;q2<Q;q2++){
			int numlinks;
			if (q1!=q2) numlinks=int(pab[q1][q2]*na[q1]*na[q2]);
			else numlinks=int(pab[q1][q2]*na[q1]*(na[q1]-1)/2);
			for(int i=0;i<numlinks;i++){
				bool bad=false;
				int v1,v2;
				do{
					bad=false;
					v1=groups_true[q1][(int)(FRANDOM*groups_true[q1].size())];
					v2=groups_true[q2][(int)(FRANDOM*groups_true[q2].size())];
					if (v1==v2) bad=true;
					// CHECK if the edge already exists
					if(!bad){
						for (int p=0;p<graph_neis[v1].size();p++){
							if (graph_neis[v1][p]==v2) {
								bad=true;
								break;
							}
						}
					}
					if(!bad){
						for (int p=0;p<graph_neis[v2].size();p++){
							if (graph_neis[v2][p]==v1) {
								bad=true;
								break;
							}
						}
					}
				}while(bad);
				graph_neis[v1].push_back(v2);
				graph_neis[v2].push_back(v1);
				vector<int> tmpvec; tmpvec.clear();
				tmpvec.push_back(v1);
				tmpvec.push_back(v2);
				graph_edges.push_back(tmpvec);
			}
		}
	}

	int num_link_tot=0;
	for (int i=0;i<N;i++) num_link_tot+=graph_neis[i].size();
	num_link_tot/=2;
	M=graph_edges.size();
	assert(M==num_link_tot&&"inconsistent number of edges, check it!");
	cout<<N<<" nodes "<<groups_true.size()<<" groups, "<<graph_edges.size()<<" edges, c="<<2.0*M/N<<flush;
	graph_max_degree=-50;
	for(int i=0; i<N; i ++)  if(int(graph_neis[i].size()) >= graph_max_degree) graph_max_degree=graph_neis[i].size();
	cout<<" max_degree="<<graph_max_degree<<endl;
}
//}}}
//{{{void blockmodel:: graph_write_gml(char *fname)
void blockmodel:: graph_write_gml(const char *fname)
{
	ofstream fout(fname);
	assert(fout.good() && "can not write to the file");
	cout<<"Writting graph into "<<fname<<"..."<<flush;
	fout << "graph [" << endl;
	for(int i=0;i<N;i++){
		fout << "  node" << endl;
		fout << "  [" << endl;
		fout << "    id " << i << endl;
//		fout << "    value " << conf_true[i] << endl;
		fout << "    label " << conf_infer[i] << endl;
		fout << "  ]"<< endl;
	}
	for(int i=0;i<M;i++){
		fout << "  edge" << endl;
		fout << "  [" << endl;
		fout << "    source " << graph_edges[i][0] << endl;
		fout << "    target " << graph_edges[i][1] << endl;
		fout << "  ]" << endl;
	}
	fout << "]" << endl;
	fout.close();
	cout<<"done."<<endl;
}
void blockmodel:: graph_write_csv(const char *fname, double energy)
{
	ofstream fout(fname);
	assert(fout.good() && "can not write to the file");
	fout << "node_id" << "," << "block" <<","<< "energy" << endl;
	for(int i=0;i<N;i++){
		fout << i << "," << conf_infer[i] << ","<<energy<<endl;
	}
	fout.close();
}
//}}}
//{{{void blockmodel:: graph_write_spm(char *fname)
void blockmodel:: graph_write_spm(const char *fname)
{
	ofstream fout(fname);
	assert(fout.good() && "can not write to the file");
	cout<<"Writting graph into "<<fname<<"..."<<flush;
	for(int i=0;i<M;i++){
		fout<<graph_edges[i][0]<<" ";
		fout<<graph_edges[i][1]<<endl;
	}
	fout.close();
	cout<<"done."<<endl;
}
//}}}
//{{{void blockmodel:: graph_write_A(char *fname)
void blockmodel:: graph_write_A(const char *fname)
{
	ofstream fout(fname);
	assert(fout.good() && "can not write to the file");
	vector< vector <int> > A;//adjacent matrix
	A.resize(N);
	for(int i=0;i<N;i++) A[i].resize(N);
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++) A[i][j]=0;
	}
	for(int i=0;i<M;i++){
		A[graph_edges[i][0]][graph_edges[i][1]]=1;
		A[graph_edges[i][1]][graph_edges[i][0]]=1;
	}
	cout<<"Writting graph into "<<fname<<"..."<<flush;
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++) fout<<A[i][j]<<" ";
		fout<<endl;
	}
	fout.close();
	cout<<"done."<<endl;
}
//}}}
//block model
//{{{void blockmodel:: bm_allocate()
void blockmodel:: bm_allocate()
{
	//need N, Q, Q_true to allocate structures na cab.
	na.resize(Q);  // size of communities (parameter of H)
	na_expect.resize(Q); // size of communities update at each timestep
	nna_expect.resize(Q); //  normalized size of expected communities update at each timestep
	nna.resize(Q); //  normalized size of expected communities update at each timestep
	na_true.resize(Q_true); //size of communities
	eta.resize(Q);
	logeta.resize(Q);

	cab.resize(Q);
	cab_true.resize(Q_true);
	logcab.resize(Q);
	pab.resize(Q);
	pab_true.resize(Q_true);
	cab_expect.resize(Q);

	for (int q=0;q<Q;q++){
		cab[q].resize(Q);
		logcab[q].resize(Q);
		pab[q].resize(Q);
		cab_expect[q].resize(Q);
	}
	for(int q=0;q<Q_true;q++){
		cab_true[q].resize(Q_true);
		pab_true[q].resize(Q_true);
	}
}
//}}}
//{{{void blockmodel:: bm_init(vector<double> pa_, vector<double > cab_)
void blockmodel::bm_init(vector<double> pa_, vector<double > cab_)
{
	// Init values for na[Q], eta[Q], pab[Q][Q] and cab[Q][Q] from pa_ and cab_
	assert((pa_.size()==Q-1 || pa_.size()==Q ) &&"there is something wrong in setting pa, are you sure you set the pa and cab properly?");
	assert((cab_.size()==Q*Q || cab_.size()==Q*(Q+1)/2 ) &&"there is something wrong in setting cab, are you sure you set the pa and cab properly.");
	int tot_size=0;
	for (int q=0;q<Q-1;q++){
		double prop_size=pa_[q];
		na[q]=prop_size*N;
		tot_size+=na[q];
		eta[q]=1.0*na[q]/N;
	}
	na[Q-1]=N-tot_size;
	eta[Q-1]=1.0*na[Q-1]/N;
	for(int q=0;q<Q;q++) logeta[q]=log(eta[q]);
	if(cab_.size() == Q*(Q+1)/2){
		int num=0;
		for (int i=0;i<Q;i++){
			for (int j=i;j<Q;j++){
				cab[i][j]=cab_[num++];
				cab[j][i]=cab[i][j];
				logcab[i][j]=log(cab[i][j]);
				logcab[j][i]=log(cab[j][i]);
			}
		}
	}else{
		int num=0;
		for (int i=0;i<Q;i++){
			for (int j=0;j<Q;j++){
				cab[i][j]=cab_[num++];
				cab[j][i]=cab[i][j];
				logcab[i][j]=log(cab[i][j]);
				logcab[j][i]=log(cab[j][i]);
			}
		}
	}
	for (int i=0;i<Q;i++){
		for (int j=0;j<Q;j++) pab[i][j]=cab[i][j]/N;
	}
	bm_show_na_cab();
}
//}}}
//{{{void blockmodel:: bm_init_random()
void blockmodel::bm_init_random()
{
	// Init values for na[Q], eta[Q], pab[Q][Q] and cab[Q][Q] randomly
	vector <double> pa_;
	for(int q=0;q<Q;q++) pa_.push_back(1.0/Q);
	int tot_size=0;
	for (int q=0;q<Q-1;q++){
		na[q]=pa_[q]*N;
		tot_size += na[q];
		eta[q]=1.0*na[q]/N;
	}
	na[Q-1]=N-tot_size;
	eta[Q-1]=1.0*na[Q-1]/N;
	for(int q=0;q<Q;q++) logeta[q]=log(eta[q]);

	vector < double > cab_;
	for(int q=0;q<(Q+1)*Q*0.5;q++) cab_.push_back(FRANDOM);
	int num=0;
	for (int i=0;i<Q;i++){
		for (int j=i;j<Q;j++){
			cab[i][j]=cab_[num++];
			cab[j][i]=cab[i][j];
		}
	}
	bm_rescale_cab();
	for (int i=0;i<Q;i++){
		for (int j=0;j<Q;j++){
			pab[i][j]=cab[i][j]/N;
			logcab[i][j]=log(cab[i][j]);
			logcab[j][i]=log(cab[j][i]);
		}
	}
	cout<<"initial matrices:"<<endl;
	bm_show_na_cab();
}
//}}}
//{{{void blockmodel:: bm_rescale_cab()
void blockmodel:: bm_rescale_cab()
{
	double cx=0.0;
	for(int q=0;q<Q;q++){
		for(int t=q;t<Q;t++) {
			if(q == t) cx += cab[q][t]*0.5/Q/Q;
			else cx += cab[q][t]/Q/Q;
		}
	}
	//cout<<"in rescaling cab, cx="<<cx<<endl;
	for(int q=0;q<Q;q++){
		for(int t=q;t<Q;t++) {
			if(bm_dc==1) cab[q][t] = 2.0*cab[q][t]/cx*graph_max_degree*N/M*N/M*graph_max_degree/4.0;
			else if(bm_dc==2) cab[q][t] = 2.0*N/M*N/M*cab[q][t]/cx/4.0;
			else cab[q][t] = 2.0*M/N*cab[q][t]/cx;
			cab[t][q] = cab[q][t];
		}
	}
}
//}}}
//{{{void blockmodel:: bm_init_uniform()
void blockmodel:: bm_init_uniform()
{
	// Uniformly init values for na[Q], eta[Q], pab[Q][Q] and cab[Q][Q]
	// note that N and Q must be set earlier
	cout<<"average size of community is "<<round(1.0/Q*N)<<endl;
	int tot_size=0;
	for (int q=0;q<Q-1;q++){
		na[q]=round(1.0*N/Q);
		//cout<<"size of community "<<q<<" -> "<<na[q]<<endl;
		tot_size+=na[q];
	}
	na[Q-1]=N-tot_size;
	eta.resize(Q,1);
	logeta.resize(Q,1);
	for(int i=0;i<Q-1;i++){
		eta[i]=(double)(na[i])/N;
		logeta[i] = log(eta[i]);
		eta[Q-1]-=eta[i];
	}
	logeta[Q-1]=log(eta[Q-1]);
	//cout<<"size of community "<<Q-1<<" -> "<<na[Q-1]<<endl;
	int num=0;

	for (int i=0;i<Q;i++){
		for (int j=i;j<Q;j++){
			if(i==j) pab[i][j]=5.0*M/N/N;
			else pab[i][j] = 1.0*M/N/N;
			pab[j][i] = pab[i][j];
		}
	}

	for (int i=0;i<Q;i++){
		for (int j=0;j<Q;j++){
			//if(i==j) cab[i][j]=2.0*pab[i][j]*N;
			//else cab[i][j]=pab[i][j]*N;
			cab[i][j]=pab[i][j]*N;
			logcab[i][j]=log(cab[i][j]);
		}
	}
}
//}}}
//{{{void blockmodel:: bm_show_na_cab()
void blockmodel:: bm_show_na_cab()
{
	cout<<"#Vector_na:"<<endl;
	for(int q=0;q<Q;q++) cout<<eta[q]<<" ";
	cout<<endl;
	cout<<"#Matrix_cab:"<<endl;
	for (int q=0;q<Q;q++){
		for (int t=0;t<Q;t++) cout<<cab[q][t]<<" ";
		cout<<endl;
	}
}
//}}}
//{{{void blockmodel:: bm_show_na_cab(string fname)
void blockmodel:: bm_show_na_cab(string fname)
{
	ofstream fout(fname.c_str());
	assert(fout.good() && "I can not open file to write marginals.");
	fout<<"#Vector_na:"<<endl;
	for(int q=0;q<Q;q++) fout<<eta[q]<<" ";
	fout<<endl;
	fout<<"#Matrix_cab:"<<endl;
	for (int q=0;q<Q;q++){
		for (int t=0;t<Q;t++) fout<<cab[q][t]<<" ";
		fout<<endl;
	}
	cout<<"na,cab have been written into "<<fname<<endl;
}
//}}}
//{{{void blockmodel:: bm_build_na_cab_from_conf()
void blockmodel:: bm_build_na_cab_from_conf()
{
	cout<<"building na and cab from configuration..."<<flush;
	for(int q=0;q<Q;q++) {
		na[q]=0;
		for(int t=0;t<Q;t++) cab[q][t]=0;
	}
	if(bm_dc==1) for(int q=0;q<Q;q++) nna[q]=0.0;
	else if(bm_dc==2) for(int q=0;q<Q;q++) nna[q]=0.0;
	for(int i=0;i<N;i++) {
		na[conf_infer[i]] += 1.0;
		if(bm_dc==1) nna[conf_infer[i]] += graph_di[i];
		else if(bm_dc==2) nna[conf_infer[i]] += graph_di[i];
	}
	for(int e=0;e<M;e++) {
		int i=graph_edges[e][0];
		int j=graph_edges[e][1];
		cab[conf_infer[i]][conf_infer[j]] += 1.0;
		if(i!=j) cab[conf_infer[j]][conf_infer[i]] += 1.0;//now cab is the number of edges between group a and b
	}
	for (int q=0;q<Q;q++) {
		eta[q]=1.0*na[q]/N;
		logeta[q]=log(eta[q]);
		for(int t=0;t<Q;t++){
			if(bm_dc==1) cab[q][t]=cab[q][t]/nna[q]/nna[t]*N;
			else if(bm_dc==2) cab[q][t]=cab[q][t]/nna[q]/nna[t]*N;
			else cab[q][t]=cab[q][t]/na[q]/na[t]*N;
			pab[q][t]=cab[q][t]/N;
			logcab[q][t]=log(cab[q][t]);
			logcab[t][q]=log(cab[q][t]);
		}
	}
	cout<<"done."<<endl;
	cout<<"initial matrices:"<<endl;
	bm_show_na_cab();
}
//}}}

//random number generator
//{{{unsigned blockmodel::init_rand4init(void)
unsigned blockmodel::init_rand4init(void)
{
	unsigned long long y;
	y = (seed*16807LL);
	seed = (y&0x7fffffff) + (y>>31);
	if (seed&0x80000000) seed = (seed&0x7fffffff) + 1;
	return seed;
}
//}}}
//{{{void blockmodel:: init_random_number_generator(void)
void blockmodel:: init_random_number_generator(int seed_)
{
	seed=seed_;
	ip=128;
	ip1=ip-24;
	ip2=ip-55;
	ip3=ip-61;
	for (unsigned i=ip3; i<ip; i++) ira[i] = init_rand4init();
}
//}}}
//compute part
//{{{double blockmodel:: compute_overlap()
double blockmodel:: compute_overlap()
{
	double max_ov=-1.0,connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na_expect[q])/N,2);
	for (int pe=0;pe<perms.size();pe++){
		double ov=0.0;
		for (int i=0;i<N;i++) ov+=(real_psi[i][perms[pe][conf_true[i]]]);
		ov/=N;
#ifdef OVL_NORM
		ov=(ov-connect)/(1-connect);
#endif
		if (ov>max_ov) max_ov=ov;
	}
	return max_ov;
}
//}}}
//{{{double blockmodel:: compute_overlap_EA()
double blockmodel:: compute_overlap_EA()
{
	double ov=0.0;
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++){
			ov += real_psi[i][q]*real_psi[i][q];
		}
	}
	ov /= N;
	double connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na_expect[q])/N,2);
#ifdef OVL_NORM
	ov=(ov-connect)/(1-connect);
#endif
	return ov;
}
//}}}
//{{{double blockmodel:: compute_overlap_fraction()
double blockmodel:: compute_overlap_fraction()
{
	compute_argmax_conf();
	return compute_config_ovl();
}
//}}}
//{{{double blockmodel:: compute_argmax_energy()
double blockmodel:: compute_argmax_energy()
{
	compute_argmax_conf();
	vector <double> term;
	term.resize(Q); for(int q=0;q<Q;q++) term[q]=0.0;
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++) {
			if(bm_dc == 2) term[q] -= graph_di[i]*pab[q][conf_infer[i]];
			else if(bm_dc == 1) term[q] -= graph_di[i]*pab[q][conf_infer[i]];
			else term[q] += log(1.0-pab[q][conf_infer[i]]);
		}
	}
	double energy=0.0;
	for(int i=0;i<N;i++){
		int qi=conf_infer[i];
		energy += log(eta[qi]);//contribution of pa
		for(int indexji=0;indexji<graph_neis[i].size();indexji++){ //contribution of edges
			int j=graph_neis[i][indexji];
			int qj=conf_infer[j];
			if(bm_dc == 2) energy += 0.5*log(graph_di[i]*graph_di[j]*pab[qi][qj]/(1.0+graph_di[i]*graph_di[j]*pab[qi][qj]));
			else if(bm_dc == 1) energy += log(graph_di[i]*graph_di[j]*cab[qi][qj])*0.5;
			else energy += log(cab[qi][qj])*0.5;
		}
		if(bm_dc==2) energy += 0.5*graph_di[i]*term[qi];
		else if(bm_dc==1) energy += 0.5*graph_di[i]*term[qi];
		else energy += 0.5*term[qi]; //contribution of non-edges
	}
	if(bm_dc == 2) return -1.0*(energy+M*log(N))/N;
	else return -1.0*energy/N;
}
//}}}
//{{{double blockmodel:: compute_config_ovl()
double blockmodel:: compute_config_ovl()
{
	double max_ov=-1.0;
	for(int pe=0;pe<perms.size();pe++){
		double ov=0.0;
		for(int j=0;j<N;j++){
			if(conf_true[j]==perms[pe][conf_infer[j]]) ov += 1.0;
		}
		ov /= N;
		if (ov>max_ov) max_ov=ov;
	}
	double connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na_expect[q])/N,2);
#ifdef OVL_NORM
	max_ov=(max_ov-connect)/(1-connect);
#endif
	return max_ov;
}
//}}}
//{{{double void:: compute_argmax_conf()
void blockmodel:: compute_argmax_conf()
{
	groups_infer.resize(Q);
	for(int q=0;q<Q;q++) groups_infer[q].clear();
	for(int i=0;i<N;i++){
		argmax_marginals[i]=-0.1;
		for(int q=0; q<Q; q++){
			if(real_psi[i][q]>argmax_marginals[i]){
				argmax_marginals[i]=real_psi[i][q];
				conf_infer[i]=q;
			}
		}
		groups_infer[conf_infer[i]].push_back(i);
	}
//	for(int q=0;q<Q;q++) cout<<groups_infer[q].size()<<" ";cout<<endl;
}
//}}}
//{{{double void:: compute_conductance()
void blockmodel:: compute_conductance()
{
//	for(int q=0;q<Q;q++) cout<<groups_infer[q].size()<<" ";cout<<endl;
	compute_argmax_conf();
	bm_conductance.resize(Q);
	assert(groups_infer.size() == Q && conf_infer.size()==N && "set argmax configuration first!");
	for(int q=0;q<Q;q++){
		int dcross=0,dtotal=0;
		if(groups_infer[q].size()==0 || groups_infer[q].size() == N){
			bm_conductance[q]=1;
			continue;
		}
		for(unsigned int idxi=0;idxi<groups_infer[q].size();idxi++){
			int i=groups_infer[q][idxi];
			dtotal += graph_neis[i].size();
			for(int idxij=0;idxij<graph_neis[i].size();idxij++){
				int j=graph_neis[i][idxij];
				if(conf_infer[j] != q) dcross ++;
			}
		}
//		cout<<"dcross="<<dcross<<" dtotal="<<dtotal<<endl;
		dtotal = min(dtotal,2*M-dtotal);
		double cond=1.0*dcross/dtotal;
		bm_conductance[q]=cond;
	}
//	cout<<"2M="<<2*M<<endl;
	cout<<"#conductance:"<<endl;
	graph_min_conductance=1.0;
	for(int q=0;q<Q;q++) {
		if(bm_conductance[q]<graph_min_conductance) graph_min_conductance=bm_conductance[q];
		cout<<bm_conductance[q]<<" ";
	}
	cout<<endl;
}
//}}}
//{{{double blockmodel:: compute_overlap_marg()
double blockmodel:: compute_overlap_marg()
{
	compute_argmax_conf();
	double ov=0.0;
	for(int i=0;i<N;i++) ov += argmax_marginals[i];
	ov /= N;
	double connect=0.0;
	for (int q=0;q<Q;q++) connect+=pow((double)(na[q])/N,2);
#ifdef OVL_NORM
	ov=(ov-connect)/(1-connect);
#endif
	return ov;
}
//}}}
//{{{double blockmodel:: compute_na_expect()
double blockmodel:: compute_na_expect()
{
	for (int j=0;j<Q;j++) {
		na_expect[j]=0.0;
		if(bm_dc==1) nna_expect[j]=0.0;
		else if(bm_dc==2) nna_expect[j]=0.0;
	}
	for (int i=0;i<N;i++){
		for (int q=0;q<Q;q++) {
			na_expect[q]+=real_psi[i][q];
			if(bm_dc==1) nna_expect[q] += graph_di[i]*real_psi[i][q];
			else if(bm_dc==2) nna_expect[q] += graph_di[i]*real_psi[i][q];
		}
	}
	return 1.0;
}
//}}}
//{{{double blockmodel::compute_log_likelihood(double f) {
double blockmodel::compute_log_likelihood(double f) {
	if(bm_dc==2) return -1.0*f*N;
	else return -1.0*f*N-M*log(1.0*N);
}
//}}}
//BP part
//{{{void blockmodel:: bp_allocate()
void blockmodel:: bp_allocate()
{
	conf_infer.resize(N);
	argmax_marginals.resize(N);
	psi.resize(N) ;
	for(int i=0;i<N;i++){
		psi[i].resize(graph_neis[i].size());
		for (int idxij=0;idxij<graph_neis[i].size();idxij++) psi[i][idxij].resize(Q);
	}
	pom_psi.resize(Q);
	psii.resize(Q);

	real_psi.resize(N) ;
	h.resize(Q);
	exph.resize(Q,0);
	for (int i=0;i<N;i++) real_psi[i].resize(Q);
	psii_iter.resize(Q);
	maxpom_psii_iter.resize(graph_max_degree);
	for (int q=0;q<Q;q++) {
		psii_iter[q].resize(graph_max_degree);
	}
	field_iter.resize(graph_max_degree);
	normtot_psi.resize(graph_max_degree);
}
//}}}
//{{{double blockmodel:: bp_compute_f()
double blockmodel:: bp_compute_f()
{
	//bp_init_h();
	compute_na_expect();
	double f_site=0;
	for (int i=0;i<N;i++){
		double di=graph_di[i];
		double maxpom_psi=-1000000.0;
		for(int j=0;j<graph_neis[i].size();j++) normtot_psi[j]=0;
		for (int q=0;q<Q;q++){
			double a=0.0;
			for (int l=0;l<graph_neis[i].size();l++){
				double b=0;
				for (int t=0;t<Q;t++) {
					if(bm_dc==1) b += di*graph_di[graph_neis[i][l]]*cab[t][q]*psi[i][l][t];//sum over messages from l -> i
					else if(bm_dc==2){
						double tmp=di*graph_di[graph_neis[i][l]]*pab[t][q];
						b += tmp/(1.0+tmp)*psi[i][l][t];//sum over messages from l -> i
					}else b += cab[t][q]*psi[i][l][t];//sum over messages from l -> i
				}
				a += log(b);
			}
			if(bm_dc==1) pom_psi[q] = a+logeta[q]-di*h[q]/N;
			else if(bm_dc==2) pom_psi[q] = a+logeta[q]-di*h[q]/N;
			else pom_psi[q]=a+logeta[q]-h[q]/N;
			if(pom_psi[q] > maxpom_psi) maxpom_psi = pom_psi[q];
		}
		double normtot=0;
		for(int q=0;q<Q;q++) normtot += exp(pom_psi[q]-maxpom_psi);
		normtot = maxpom_psi+log(normtot);
		f_site += normtot;
	}
	f_site/=N;

	double f_link=0;
	for (int i=0;i<N;i++){
		double di=graph_di[i];
		for (int l=0;l<graph_neis[i].size();l++){
			double dl=graph_di[graph_neis[i][l]];
			double norm_L=0;
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			for (int q1=0;q1<Q;q1++){
				for (int q2=q1;q2<Q;q2++){
					if(q1==q2){
						if(bm_dc==1) norm_L+=di*dl*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]);
						//if(bm_dc==2) norm_L+=di*dl*pab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]);
						else if(bm_dc==2){
							double tmp=di*dl*pab[q1][q2];
							norm_L+=tmp/(1.0+tmp)*(psi[i][l][q1]*psi[i2][l2][q2]);
						}else norm_L+=cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]);
					}else{
						if(bm_dc==1) norm_L+=di*dl*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
						//else if(bm_dc==2) norm_L+=di*dl*pab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
						else if(bm_dc==2) {
							double tmp=di*dl*pab[q1][q2];
							norm_L+=tmp/(1.0+tmp)*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
						}else norm_L+=cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
					}
				}
			}
			f_link+=log(norm_L);
		}
	}
	f_link/=2*N;

	double last_term=0;
	for (int q1=0;q1<Q;q1++){
		for (int q2=q1;q2<Q;q2++){
			if (q1==q2){
				if(bm_dc==1) last_term+=0.5*cab[q1][q2]*nna_expect[q1]/N*nna_expect[q2]/N;
				else if(bm_dc==2) last_term+=0.5*pab[q1][q2]*nna_expect[q1]*nna_expect[q2]/N;
				else last_term+=0.5*cab[q1][q2]*na_expect[q1]/N*na_expect[q2]/N;
			}
			else {
				if(bm_dc==1) last_term+=cab[q1][q2]*nna_expect[q1]/N*nna_expect[q2]/N;
				else if(bm_dc==2) last_term+=pab[q1][q2]*nna_expect[q1]*nna_expect[q2]/N;
				else last_term+=cab[q1][q2]*na_expect[q1]/N*na_expect[q2]/N;
			}
		}
	}
	double totalf=f_site-f_link+last_term;
//	cout<<"f_site="<<f_site<<" f_lin="<<f_link<<" last_term="<<last_term<<endl;
	if(bm_dc==2) totalf += 1.0*M/N*log(1.0*N);
	return -totalf;
}
//}}}
//{{{void blockmodel:: bp_init_h()
void blockmodel:: bp_init_h()
{
	h.resize(Q,0);
	for(int q=0;q<Q;q++) h[q]=0;
	for (int i=0;i<N;i++){
		// update h
		for (int q1=0;q1<Q;q1++){
			for (int q2=0;q2<Q;q2++) {
				if(bm_dc==1) h[q1] += graph_di[i]*cab[q2][q1]*real_psi[i][q2];//degree corrected model
				else if(bm_dc==2) h[q1] += graph_di[i]*cab[q2][q1]*real_psi[i][q2];//degree corrected model
				else h[q1]+=cab[q2][q1]*real_psi[i][q2];
			}
		}
	}
	for(int q=0;q<Q;q++)  exph[q]=myexp(-h[q]/N);
}
//}}}
//{{{void blockmodel::bp_init_m(double pp)
void blockmodel::bp_init_m(double pp)
{
	//initialize BP messages with magnetization close to planted conf_true controled by pp
	for (int i=0;i<N;i++){
		for(int q=0;q<Q;q++) {
			if(q==conf_true[i]) real_psi[i][q] = pp+(1.0-pp)*FRANDOM;
			else real_psi[i][q]=FRANDOM*(1.0-pp);
		}
		for (int idxij=0;idxij<graph_neis[i].size();idxij++) {
			for(int q=0;q<Q;q++) {
				if(q==conf_true[i]) psi[i][idxij][q] = pp+(1.0-pp)*FRANDOM;
				else psi[i][idxij][q] = FRANDOM*(1.0-pp);
			}
		}
	}
	bp_init_h();
}
//}}}
//{{{void blockmodel:: bp_init(int initflag)
void blockmodel:: bp_init(int init_flag)
{
	for (int i=0;i<N;i++){
		if(init_flag==1){//random intialize messages
			double norm=0.0;
			for(int q=0;q<Q;q++) {
				real_psi[i][q] = FRANDOM;
				norm += real_psi[i][q];
			}
			for(int q=0;q<Q;q++) real_psi[i][q] /= norm;
			for (int idxij=0;idxij<graph_neis[i].size();idxij++) {
				int j=graph_neis[i][idxij];
				int idxji=graph_neis_inv[i][idxij];
				norm=0.0;
				for(int q=0;q<Q;q++) {
					psi[j][idxji][q] = FRANDOM;
					norm += psi[j][idxji][q];
				}
				for(int q=0;q<Q;q++) psi[j][idxji][q] /= norm;

//				norm=0.0;
//				for(int q=0;q<Q;q++) {
//					psi[i][idxij][q] = FRANDOM;
//					norm += psi[i][idxij][q];
//				}
//				for(int q=0;q<Q;q++) psi[i][idxij][q] /= norm;
			}
		}else if(init_flag == 2){//initialize messages by planted conf_true
			for(int q=0;q<Q;q++) {
				if(q==conf_true[i]) real_psi[i][q] = 1.0;
				else real_psi[i][q]=0.000;
			}
			for (int idxij=0;idxij<graph_neis[i].size();idxij++) {
				int j=graph_neis[i][idxij];
				int idxji=graph_neis_inv[i][idxij];
				for(int q=0;q<Q;q++) {
					if(q==conf_true[i]) psi[j][idxji][q] = 1.0;
					else psi[j][idxji][q] = 0.0;
				}
			}
		}else if(init_flag == 0){ //do not reinit, should be used in learning
		}
	}
	if(vflag > 1){
		if(init_flag==2) cout<<"init from true configuration"<<endl;
		cout<<"after initialization, overlap="<<compute_overlap_fraction()<<endl;
	}
	bp_init_h();
}
//}}}
//{{{double blockmodel::bp_iter_update_psi(int i,double dumping_rate)
double blockmodel::bp_iter_update_psi(int i,double dumping_rate)
{
	int l,q;
	double di=graph_di[i];
	double a=1.0, b=0.0, normtotal=0.0, normtot_real=0.0;
	for(int j=0;j<graph_neis[i].size();j++) normtot_psi[j]=0;
	for (q=0;q<Q;q++){
		a=1.0;
		for (l=0;l<graph_neis[i].size();l++){
			b=0.0;
			for (int t=0;t<Q;t++) {
				if(bm_dc==1) b += di*graph_di[graph_neis[i][l]]*cab[t][q]*psi[i][l][t];//sum over messages from l -> i
				else if(bm_dc==2) {
					double tmp=di*graph_di[graph_neis[i][l]]*pab[t][q];
					b += tmp/(1.0+tmp)*psi[i][l][t];//sum over messages from l -> i
				}else b += cab[t][q]*psi[i][l][t];//sum over messages from l -> i
			}
			a*=b;
			field_iter[l]=b;
		}
		if(bm_dc==1) pom_psi[q] = a*eta[q]*exp(-1.0*di*h[q]/N);
		else if(bm_dc==2) pom_psi[q] = a*eta[q]*exp(-1.0*di*h[q]/N);
		else pom_psi[q]=a*eta[q]*exph[q];
		normtot_real += pom_psi[q];
		for (l=0;l<graph_neis[i].size();l++){
			if(field_iter[l]<EPS){//to cure the problem that field_iter[l] may be very small
				double tmprob=1.0;
				for(int lx=0;lx<graph_neis[i].size();lx++){
					if(lx==l) continue;
					tmprob *= field_iter[lx];
				}
				psii_iter[q][l]=tmprob;
			}else psii_iter[q][l]=pom_psi[q]/(field_iter[l]);
			normtot_psi[l]+= psii_iter[q][l];
		}
	}
	for (int q1=0;q1<Q;q1++){
		for (int q2=0;q2<Q;q2++) {
			if(bm_dc==1) h[q1] -= di*cab[q2][q1]*real_psi[i][q2];
			else if(bm_dc==2) h[q1] -= di*cab[q2][q1]*real_psi[i][q2];
			else h[q1] -= cab[q2][q1]*real_psi[i][q2];
		}
	}
	// normalization
	double mymaxdiff=-100.0;
	for (q=0;q<Q;q++){
		real_psi[i][q]=pom_psi[q]/normtot_real;
		for (l=0;l<graph_neis[i].size();l++){
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double mydiff=fabs(psi[i2][l2][q]-psii_iter[q][l]/normtot_psi[l]);
			//mymaxdiff += mydiff;
			if(mydiff > mymaxdiff) mymaxdiff = mydiff;
			psi[i2][l2][q]=(dumping_rate)*psii_iter[q][l]/normtot_psi[l]+(1.0-dumping_rate)*psi[i2][l2][q];//update psi_{i\to j}
		}
	}
	//mymaxdiff /= Q*graph_neis[i].size();
	for (int q1=0;q1<Q;q1++){
		for (int q2=0;q2<Q;q2++) {
			if(bm_dc==1) h[q1] += di*cab[q2][q1]*real_psi[i][q2];
			else if(bm_dc==2) h[q1] += di*cab[q2][q1]*real_psi[i][q2];
			else h[q1] += cab[q2][q1]*real_psi[i][q2];
		}
	}
	for(int q=0;q<Q;q++)  exph[q]=myexp(-h[q]/N);
	return mymaxdiff;
}
//}}}
//{{{double blockmodel:: bp_iter_update_psi_large_degree(int i)
double blockmodel:: bp_iter_update_psi_large_degree(int i,double dumping_rate)
{
	int l,q;
	double di=graph_di[i];
	double mymaxdiff=-100.0;
	double a=1.0, b=0.0, normtotal=0.0, normtot_real=0.0;
	for(int j=0;j<graph_neis[i].size();j++) normtot_psi[j]=0;
	double maxpom_psi=-100000000.0;
	double xxx=-100000000.0;
	for(int j=0;j<graph_neis[i].size();j++) maxpom_psii_iter[j]=xxx;
	for (q=0;q<Q;q++){
		a=0.0;//log value
		// sum of all graphbors of i
		for (l=0;l<graph_neis[i].size();l++){
			b=0.0;
			for (int t=0;t<Q;t++) {
				if(bm_dc==1) b += di*graph_di[graph_neis[i][l]]*cab[t][q]*psi[i][l][t];//sum over messages from l -> i
				else if(bm_dc==2) {
					double tmp=di*graph_di[graph_neis[i][l]]*pab[t][q];
					b += tmp/(1.0+tmp)*psi[i][l][t];//sum over messages from l -> i
				}else b += cab[t][q]*psi[i][l][t];//sum over messages from l -> i
			}
			double tmp=log(b);
			a += tmp;
			field_iter[l]=tmp;
		}
		if(bm_dc==1) pom_psi[q] = a+logeta[q]-1.0*di*h[q]/N;
		else if(bm_dc==2) pom_psi[q] = a+logeta[q]-1.0*di*h[q]/N;
		else pom_psi[q]=a+logeta[q]-h[q]/N;

		if(pom_psi[q] > maxpom_psi) maxpom_psi = pom_psi[q];
		for (l=0;l<graph_neis[i].size();l++) {
			psii_iter[q][l]=pom_psi[q]-field_iter[l];
			if(psii_iter[q][l] > maxpom_psii_iter[l]) maxpom_psii_iter[l]=psii_iter[q][l];
		}
	}
	for(q=0;q<Q;q++){
		normtot_real += exp(pom_psi[q]-maxpom_psi);
		for(l=0;l<graph_neis[i].size();l++) normtot_psi[l] += exp(psii_iter[q][l]-maxpom_psii_iter[l]);
	}
	for (int q1=0;q1<Q;q1++){
		for (int q2=0;q2<Q;q2++) {
			if(bm_dc==1) h[q1] -= di*cab[q2][q1]*real_psi[i][q2];
			else if(bm_dc==2) h[q1] -= di*cab[q2][q1]*real_psi[i][q2];
			else h[q1] -= cab[q2][q1]*real_psi[i][q2];
		}
	}
	// normalization
	for (q=0;q<Q;q++){
		real_psi[i][q]=exp(pom_psi[q]-maxpom_psi)/normtot_real;
		for (l=0;l<graph_neis[i].size();l++){
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double thisvalue=exp(psii_iter[q][l]-maxpom_psii_iter[l])/normtot_psi[l];
			double mydiff=fabs(psi[i2][l2][q]-thisvalue);
			if(mydiff > mymaxdiff) mymaxdiff = mydiff;
			psi[i2][l2][q]=(dumping_rate)*thisvalue+(1-dumping_rate)*psi[i2][l2][q];

		}
	}
	for (int q1=0;q1<Q;q1++){
		for (int q2=0;q2<Q;q2++) {
			if(bm_dc==1) h[q1] += di*cab[q2][q1]*real_psi[i][q2];
			else if(bm_dc==2) h[q1] += di*cab[q2][q1]*real_psi[i][q2];
			else h[q1] += cab[q2][q1]*real_psi[i][q2];
		}
	}
	for(int q=0;q<Q;q++) exph[q]=myexp(-h[q]/N);
	return mymaxdiff;
}
//}}}
//{{{int blockmodel::bp_converge(double bp_err, int max_iter_time, int init_flag, double dumping_rate)
int blockmodel::bp_converge(double bp_err, int max_iter_time,int init_flag, double dumping_rate)
{
//	ofstream fout("log.txt");
	if(init_flag < 3) bp_init(init_flag);
	else{
		double prob=1.0*init_flag/100-int(1.0*init_flag/100);
		bp_init_m(prob);
	}
	if(vflag >= 10) show_marginals(vflag);
	else if(vflag >= 3) show_marginals(10);
	bp_init_h();
	int iter_time=0;
	for(iter_time=0; iter_time<max_iter_time;iter_time++){
		double maxdiffm = -100.0;
		vector <int> ranseq;
		ranseq.resize(N);
//		shuffle_seq(ranseq);
		for(int iter_inter_time=0;iter_inter_time<N;iter_inter_time++){
//			int i=ranseq[iter_inter_time];
			int i=int(FRANDOM*N);
			double diffm = 0.0;
			if(graph_neis[i].size()>=LARGE_DEGREE) {
				diffm=bp_iter_update_psi_large_degree(i,dumping_rate);
			}else{
				diffm=bp_iter_update_psi(i,dumping_rate);
			}
			if(diffm > maxdiffm) maxdiffm=diffm;
		}
//		fout<<iter_time<<" "<<maxdiffm<<" "<<compute_overlap_marg()<<" "<<bp_compute_f()<<" "<<real_psi[0][0]<<" "<<real_psi[0][1]<<endl;
		if(vflag >= 2) cout<<maxdiffm<<" "<<flush;
		else if(vflag == 1) cout<<"."<<flush;
		if(maxdiffm < bp_err ){
			if(vflag >= 1) cout<<" bp-:) iter_time="<<iter_time<<" "<<endl;
			if(vflag >= 10) show_marginals(vflag);
			else if(vflag >= 3) show_marginals(10);
			bp_last_conv_time=iter_time;
			bp_last_diff=maxdiffm;
//			fout.close();
			return iter_time;
		}
		if(vflag >= 4) show_marginals(10);
	}
	if(vflag >= 1) cout<<" bp-:( iter_time="<<max_iter_time<<" "<<endl;
	if(vflag >= 3) show_marginals(10);
//			fout.close();
	return -1;
}
//}}}
//{{{double blockmodel:: bp_compute_cab_expect()
double blockmodel:: bp_compute_cab_expect()
{
	for(int q1=0;q1<Q;q1++){
		for (int q2=0;q2<Q;q2++) cab_expect[q1][q2]=0.0;
	}
	for(int i=0;i<N;i++){
		double di=graph_di[i];
		for(int l=0;l<graph_neis[i].size();l++){
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double dl=graph_di[i2];
			double norm_L=0.0;
			for(int q1=0;q1<Q;q1++){
				for (int q2=q1;q2<Q;q2++){
					if (q1==q2){
						if(bm_dc==1) norm_L += di*dl*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]);
						else if(bm_dc==2){
							double tmp=di*dl*pab[q1][q2];
							norm_L += tmp/(1.0+tmp)*(psi[i][l][q1]*psi[i2][l2][q2]);
						}else norm_L += cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]);
					}else{
						if(bm_dc==1) norm_L += di*dl*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
						else if(bm_dc==2){
							double tmp=di*dl*pab[q1][q2];
							norm_L += tmp/(1.0+tmp)*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
						}else norm_L += cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1]);
					}
				}
			}
			for(int q1=0;q1<Q;q1++){
				for (int q2=q1;q2<Q;q2++){
					if (q1==q2){
						if(bm_dc==1) cab_expect[q1][q2] += 0.5*di*dl*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2])/norm_L;
						else if(bm_dc==2) {
							double tmp=di*dl*pab[q1][q2];
							cab_expect[q1][q2] += 0.5*tmp/(1.0+tmp)*(psi[i][l][q1]*psi[i2][l2][q2])/norm_L;
						}else cab_expect[q1][q2] += 0.5*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2])/norm_L;
					}else{
						if(bm_dc==1) cab_expect[q1][q2] += 0.5*di*dl*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1])/norm_L;
						else if(bm_dc==2) {
							double tmp=di*dl*pab[q1][q2];
							cab_expect[q1][q2] += 0.5*tmp/(1.0+tmp)*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1])/norm_L;
						}else cab_expect[q1][q2] += 0.5*cab[q1][q2]*(psi[i][l][q1]*psi[i2][l2][q2]+psi[i][l][q2]*psi[i2][l2][q1])/norm_L;
					}
				}
			}
		}
	}
	for(int q1=0;q1<Q;q1++){
		for(int q2=q1;q2<Q;q2++){
			if((na_expect[q1]>EPS) && (na_expect[q2]>EPS)){
				if(q1!=q2){
					if(bm_dc==1) cab_expect[q1][q2] *= N/(nna_expect[q1]*nna_expect[q2]);
					else if(bm_dc==2) cab_expect[q1][q2] *= N/(nna_expect[q1]*nna_expect[q2]);
					else cab_expect[q1][q2] *= N/(na_expect[q1]*na_expect[q2]);
					cab_expect[q2][q1] = cab_expect[q1][q2];
				}else{
					if(bm_dc==1) cab_expect[q1][q2] *= 2.*N/(nna_expect[q1]*nna_expect[q2]);
					else if(bm_dc==2) cab_expect[q1][q2] *= 2.*N/(nna_expect[q1]*nna_expect[q2]);
					else cab_expect[q1][q2] *= 2.*N/(na_expect[q1]*na_expect[q2]);
				}
			}
		}
	}
	double max_diff=100;
	return max_diff;
}
//}}}
//MF part
//{{{void blockmodel:: nmf_allocate()
void blockmodel:: nmf_allocate()
{
	pom_psi.resize(Q);
	conf_infer.resize(N);
	argmax_marginals.resize(N);
	real_psi.resize(N) ;
	for (int i=0;i<N;i++) real_psi[i].resize(Q);
	h.resize(Q);
}
//}}}
//{{{void blockmodel::nmf_init(int init_flag)
void blockmodel::nmf_init(int init_flag)
{
	assert(init_flag == 0 ||init_flag == 1 ||init_flag == 2);
	if(init_flag==1){//random intialize messages
		for(int i=0;i<N;i++){
			double norm=0.0;
			for(int q=0;q<Q;q++) {
				real_psi[i][q] = FRANDOM;
				norm += real_psi[i][q];
			}
			for(int q=0;q<Q;q++) real_psi[i][q] /= norm;
		}
	}else if(init_flag == 2){//initialize messages by planted conf_true
		for(int i=0;i<N;i++){
			for(int q=0;q<Q;q++)  {
				if(q==conf_true[i]) real_psi[i][q] = 0.99;
				else real_psi[i][q]=0.0001;
			}
		}
	}else if(init_flag == 0){ //do not reinit, should be used in learning
	}
	nmf_init_h();
}
//}}}
//{{{int blockmodel::nmf_converge(double nmf_err, int max_iter_time, double dumping_rate)
int blockmodel::nmf_converge(double nmf_err, int max_iter_time, int init_flag, double dumping_rate)
{
	if(init_flag < 3) nmf_init(init_flag);
	else{
		double prob=1.0*init_flag/100-int(1.0*init_flag/100);
		cout<<"prob="<<prob<<endl;
		nmf_init_m(prob);
	}
	for(int iter_time=0; iter_time<max_iter_time;iter_time++){
		double maxdiffm = -100.0;
		vector <int> ranseq;
		ranseq.resize(N);
		shuffle_seq(ranseq);
		for(int iter_inter_time=0;iter_inter_time<N;iter_inter_time++){
			//int i=int(rg->rdflt()*N);
			//int i=int(FRANDOM*N);
			int i=ranseq[iter_inter_time];
			double diffm = 0.0;
			diffm=nmf_iterate(i,dumping_rate);
			if(diffm > maxdiffm) maxdiffm=diffm;
		}
		if(vflag >= 2) cout<<maxdiffm<<" "<<flush;
		else if(vflag == 1) cout<<"."<<flush;
		if(maxdiffm < nmf_err ){
			if(vflag >= 1) cout<<" MF-:) iter_time="<<iter_time<<" "<<endl;
			if(vflag >= 3) show_marginals(10);
			return iter_time;
		}
	}
	if(vflag >= 1) cout<<" MF-:( iter_time="<<max_iter_time<<" "<<endl;
	if(vflag >= 3) show_marginals(10);
	return -1;
}
//}}}
//{{{void blockmodel:: nmf_init_h()
void blockmodel:: nmf_init_h()
{
	for(int q=0;q<Q;q++) h[q]=0.0;
	for (int i=0;i<N;i++){
		for (int q1=0;q1<Q;q1++){
			//for (int q2=0;q2<Q;q2++) h[q1]+=log(1.0-pab[q1][q2])*real_psi[i][q2];
			for (int q2=0;q2<Q;q2++) h[q1] -= pab[q1][q2]*real_psi[i][q2];
		}
	}
}
//}}}
//{{{void blockmodel::nmf_init_m(double pp)
void blockmodel::nmf_init_m(double pp)
{
	//initialize MF messages with magnetization close to planted conf_true controled by pp
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++)  {
			if(q==conf_true[i]) real_psi[i][q] = pp+(1.0-pp)*FRANDOM;
			else real_psi[i][q]=FRANDOM*(1.0-pp); }
	}
	nmf_init_h();
}
//}}}
//{{{double blockmodel::nmf_iterate(int i)
double blockmodel::nmf_iterate(int i, double dumping_rate)
{
	//double mymaxdiff=-100.0;
	double normtot=0.0;
	for (int q=0;q<Q;q++){
		pom_psi[q] = logeta[q];
		for(int idxij=0;idxij<graph_neis[i].size();idxij++){
			int j=graph_neis[i][idxij];
			//for(int t=0;t<Q;t++) pom_psi[q] += (logcab[q][t]-log(1.0-pab[q][t]))*real_psi[j][t];
			for(int t=0;t<Q;t++) pom_psi[q] += (logcab[q][t]+pab[q][t])*real_psi[j][t];
		}
		pom_psi[q]=exp(pom_psi[q]+h[q]);
		normtot += pom_psi[q];
	}

	for(int q=0;q<Q;q++){
		//for(int t=0;t<Q;t++) h[q] -= log(1.0-pab[q][t])*real_psi[i][t];
		for(int t=0;t<Q;t++) h[q] += pab[q][t]*real_psi[i][t];
	}
	// normalization
	double mymaxdiff=0.0;
	for (int q=0;q<Q;q++){
		double tmppsi=pom_psi[q]/normtot;
		double mydiff=fabs(real_psi[i][q]-tmppsi);
		real_psi[i][q]=(dumping_rate)*tmppsi+(1-dumping_rate)*real_psi[i][q];
		//if(mydiff > mymaxdiff) mymaxdiff=mydiff;
		mymaxdiff += mydiff;
	}
	mymaxdiff /= Q;
	for(int q=0;q<Q;q++){
		//for(int t=0;t<Q;t++) h[q] += log(1.0-pab[q][t])*real_psi[i][t];
		for(int t=0;t<Q;t++) h[q] -= pab[q][t]*real_psi[i][t];
	}
	return mymaxdiff;
}
//}}}
//{{{double blockmodel::nmf_compute_f()
double blockmodel::nmf_compute_f()
{
	double f=0.0;
	for(int i=0;i<N;i++){
		for(int a=0;a<Q;a++){
			double tmp=0.0;
			for(int idxij=0;idxij<graph_neis[i].size();idxij++){
				int j=graph_neis[i][idxij];
				for(int b=0;b<Q;b++) tmp += (logcab[a][b]+pab[a][b])*real_psi[j][b];
			}
			f -= (tmp+h[a])*real_psi[i][a];
		}
	}
	f *= 0.5;
	for(int i=0;i<N;i++){
		for(int a=0;a<Q;a++){
			f -= real_psi[i][a]*(logeta[a]-log(real_psi[i][a]));
		}
	}
	f /= N;
	return f;
}
//}}}
//{{{double blockmodel:: nmf_compute_cab_expect()
double blockmodel:: nmf_compute_cab_expect()
{
	for(int q1=0;q1<Q;q1++){
		for (int q2=0;q2<Q;q2++) cab_expect[q1][q2]=0.0;
	}
	for(int i=0;i<N;i++){
		for(int l=0;l<graph_neis[i].size();l++){
			int i2=graph_neis[i][l];
			int l2=graph_neis_inv[i][l];
			double norm_L=0.0;
			for(int q1=0;q1<Q;q1++){
				for (int q2=q1;q2<Q;q2++){
					if (q1==q2) norm_L += cab[q1][q2]*(real_psi[i][q1]*real_psi[i2][q2]);
					else  norm_L += cab[q1][q2]*(real_psi[i][q1]*real_psi[i2][q2]+real_psi[i][q2]*real_psi[i2][q1]);
				}
			}
			for(int q1=0;q1<Q;q1++){
				for (int q2=q1;q2<Q;q2++){
					if (q1==q2) cab_expect[q1][q2] += 0.5*cab[q1][q2]*(real_psi[i][q1]*real_psi[i2][q2])/norm_L;
					else cab_expect[q1][q2] += 0.5*cab[q1][q2]*(real_psi[i][q1]*real_psi[i2][q2]+real_psi[i][q2]*real_psi[i2][q1])/norm_L;
				}
			}
		}
	}
	for(int q1=0;q1<Q;q1++){
		for(int q2=q1;q2<Q;q2++){
			if((na_expect[q1]!=0) && (na_expect[q2]!=0)){
				if(q1!=q2){
					cab_expect[q1][q2] *= N/(na_expect[q1]*na_expect[q2]);
					cab_expect[q2][q1] = cab_expect[q1][q2];
				}else cab_expect[q1][q2] *= 2.*N/(na_expect[q1]*na_expect[q2]);
			}
		}
	}
	double max_diff=100;
	return max_diff;
}
//}}}
//Learning and Inference part
//{{{void blockmodel::EM_learning()
void blockmodel::EM_learning(int mode, int init_flag, double conv_crit, int time_conv, double dumping_rate, double learning_rate,int learning_max_time, double learning_conv_crit, bool no_learn_pa)
{
	if(mode==1) {
		bp_allocate();
		bp_init(init_flag);
		cout<<"BP learning"<<endl;
	}else {
		nmf_allocate();
		bp_init(init_flag);
		cout<<"MF learning"<<endl;
	}
	double fold=0.0,ovl=0.0;
	if(vflag >= 0) {
		cout<<"#t\tfree_energy"<<deli<<"argmax overlap"<<deli;
		if(groups_true.size() > 0) cout<<"fraction"<<deli;
		cout<<"delta_infer"<<deli<<"bp_conv_crit"<<endl;
		//cout<<"delta_infer"<<deli<<endl;
	}
	int learning_time=0;
	double fdiff=1.0;
	for(learning_time=0;learning_time<learning_max_time;learning_time++){
		double fnew=0.0;
		if(mode==1) {
			if(fdiff < conv_crit) conv_crit *= 0.1;
	//		bp_init(init_flag);
			bp_converge(conv_crit,time_conv,0, dumping_rate);
			compute_na_expect();
			bp_compute_cab_expect();
			fnew=bp_compute_f();
		}else{
			nmf_converge(conv_crit,time_conv,0, dumping_rate);
			compute_na_expect();
			nmf_compute_cab_expect();
			fnew=nmf_compute_f();
		}
		fdiff=fabs(fnew-fold);
		fold=fnew;
		ovl=compute_overlap_marg();
		if(vflag >= 0) {
			cout<<"#"<<learning_time<<"\t "<<fnew<<deli<<ovl<<deli;
			if(groups_true.size() > 0) cout<<compute_overlap_fraction()<<deli;
			cout<<fdiff<<deli<<conv_crit<<deli;
			//cout<<fdiff<<deli;
		}
		if(isnan(fold) || isinf(fold)) break;
		if(vflag >= 0) { for (int q=0;q<Q;q++) cout<<(double)(na_expect[q])/(double)(N)<<" "; cout<<endl; }
		if(fdiff<learning_conv_crit) break;
		EM_step(learning_rate,no_learn_pa);
	}
	cout<<"num_iteration_learning="<<learning_time<<endl;
	bp_init(init_flag);
	bp_converge(conv_crit,time_conv,0, dumping_rate);
	compute_na_expect();
	bp_compute_cab_expect();
	output_f_ovl(mode);
}
//}}}
//{{{void blockmodel::walk_learning()
void blockmodel::walk_learning(int mode, int init_flag, double conv_crit, int time_conv, double dumping_rate, int learning_max_time,int wblock ,double wstep, bool no_learn_pa)
{
	if(mode==1) {
		bp_allocate();
		bp_init(init_flag);
		cout<<"BP learning"<<endl;
	}else {
		nmf_allocate();
		bp_init(init_flag);
		cout<<"MF learning"<<endl;
	}
	double fold=100.0,ovl=0.0;
	double delta=wstep;
	cout<<"wtime="<<learning_max_time<<" wblock="<<wblock<<" wstep="<<delta<<endl;
	int learning_time=0,wtime=0;
	for(learning_time=0;learning_time<learning_max_time;learning_time++){
		int a=int(FRANDOM *Q);
		int b=int(FRANDOM *Q);
		double value=cab[a][b];
		cab[a][b] += FRANDOM>0.5?delta:-1.0*delta;
		cab[b][a] = cab[a][b];
		double fnew=0.0;
		if(mode==1) {
			bp_converge(conv_crit,time_conv,0, dumping_rate);
			compute_na_expect();
			bp_compute_cab_expect();
			fnew=bp_compute_f();
		}else{
			nmf_converge(conv_crit,time_conv,0, dumping_rate);
			compute_na_expect();
			nmf_compute_cab_expect();
			fnew=nmf_compute_f();
		}
		if(fnew > fold || isnan(fnew) || isinf(fnew)) {
			if(wtime > wblock) {
				cout<<" blocked"<<endl;
				break;
			}
			cab[a][b] = value;
			cab[b][a] = cab[a][b];
			wtime++;
		}else {
			if(vflag >= 0) cout<<fnew<<" "<<flush;
			if(groups_true.size() > 0) cout<<"("<<compute_overlap_fraction()<<") "<<deli;
			fold=fnew;
			wtime=0;
		}
	}
	if(vflag >= 0) cout<<"time out."<<endl;
	output_f_ovl(mode);
}
//}}}
//{{{void blockmodel::do_inference(int mode)
void blockmodel::do_inference(int mode,int init_flag,double conv_crit, int time_conv, double dumping_rate)
{
	int niter=0;
//	assert(init_flag == 0 ||init_flag == 1 ||init_flag == 2);
	if(mode==1) {
		bp_allocate();
		niter=bp_converge(conv_crit,time_conv, init_flag, dumping_rate);
	}else{
		nmf_allocate();
		niter=nmf_converge(conv_crit,time_conv,init_flag, dumping_rate);
	}
	cout<<"num_iterations="<<niter<<endl;
	output_f_ovl(mode);
}
//}}}
//{{{void blockmodel::EM_step()
void blockmodel::EM_step(double learning_rate,bool no_learn_pa)
{
	//graded descent to learn eta, cab and pab from expected values
	if(!no_learn_pa){
		for(int q=0;q<Q;q++){
			na[q] = learning_rate*na_expect[q]+(1.0-learning_rate)*na[q];
			eta[q]= 1.0*na[q]/N;
			logeta[q]= log(eta[q]);
		}
	}
	for(int i=0;i<Q;i++){
		for(int j=0;j<Q;j++){
			cab[i][j] = learning_rate*cab_expect[i][j]+(1.0-learning_rate)*cab[i][j];
			logcab[i][j] =log(cab[i][j]);
			pab[i][j] = cab[i][j]/N;
		}
	}
}
//}}}
//Output part
//{{{void blockmodel:: output_f_ovl(int mode)
void blockmodel:: output_f_ovl(int mode)
{
	double f=0.0;
	if(mode==1) f=bp_compute_f();
	else f=nmf_compute_f();
	double l=compute_log_likelihood(f);
	cout<<"f="<<f<<deli<<"L="<<l<<deli<<"argmax_ovl="<<compute_overlap_marg()<<deli;
	if(groups_true.size() > 0 && groups_true.size()==Q) cout<<"overlap="<<compute_overlap()<<deli<<"fraction="<<compute_overlap_fraction()<<endl;
	else cout<<endl;
}
//}}}
//{{{void blockmodel:: output_marginals(string fname)
void blockmodel:: output_marginals(string fname)
{
	ofstream fout(fname.c_str());
	assert(fout.good() && "I can not open file to write marginals.");
	double f=0.0;
	f=bp_compute_f();
	double l=-1.0*f*N-M*log(1.0*N);
	fout<<"f="<<f<<deli<<"L="<<l<<deli<<"argmax_ovl="<<compute_overlap_marg()<<deli;
	if(groups_true.size() > 0) fout<<"overlap="<<compute_overlap()<<deli<<"fraction="<<compute_overlap_fraction()<<endl;
	else fout<<endl;
	compute_argmax_conf();
	fout<<"argmax_configuration:"<<endl;
	for(int i=0;i<N;i++) fout<<conf_infer[i]<<" ";
	fout<<endl;
	fout<<endl;

	fout<<"marginals:"<<endl;
	for(int i=0;i<N;i++){
		for(int q=0;q<Q;q++) fout<<real_psi[i][q]<<" ";
		fout<<endl;
	}
	fout<<endl;
	fout<<"argmax_marginals:"<<endl;
	for(int i=0;i<N;i++) fout<<argmax_marginals[i]<<" ";
	fout<<endl;
	cout<<"marginals, argmax configuration, argmax marginals are written into "<<fname<<"."<<endl;
}
//}}}
//{{{void blockmodel::show_marginals(int num)
void blockmodel::show_marginals(int num)
{
	cout<<"Marginals from node 0 to node "<<num<<":"<<endl;
	for(int i=0;i<num;i++){
		for(int q=0;q<Q;q++) cout<<real_psi[i][q]<<" "<<flush;
		cout<<endl;
	}
	cout<<endl;
//	cout<<"similarity of groups:"<<endl;
//	for(int q=0;q<Q;q++){
//		for(int t=q+1;t<Q;t++){
//			double sim=0.0;
//			for(int i=0;i<N;i++) sim += fabs(real_psi[i][q]-real_psi[i][t]);
//			sim /= N;
//			cout<<q<<t<<"="<<sim<<" ";
//		}
//		cout<<endl;
//	}

}
//}}}
//permutation
//{{{ void blockmodel::init_perms()
void blockmodel::init_perms()
{
	perms.clear();
	vector <int> myperm;
	for(int i=0;i<Q;i++) myperm.push_back(i);
	sort(myperm.begin(),myperm.end());
	if(Q>Q_PERMU){
		perms.push_back(myperm);
	}else{
		do perms.push_back(myperm);
		while(next_permutation(myperm.begin(),myperm.end()));
	}
}
//}}}
//spectral clustering part
//{{{string get_std_from_cmd(string cmd)
string get_std_from_cmd(const char* cmd)
{
	string data;
	FILE *stream;
	char buffer[1024];
	stream = popen(cmd, "r");
	while ( fgets(buffer, 1024, stream) != NULL )
	data.append(buffer);
	pclose(stream);
	return data;
}
//}}}
//{{{void blockmodel:: spec(string fname,int mode)
void blockmodel:: spec(string fname,string origin_fname,int mode)
{
	ostringstream cmd;
	if(mode != 4) cmd<<"./spc.py "<<N<<" "<<fname<<" "<<Q<<" "<<mode;
	else cmd<<"mspc.py "<<origin_fname<<" "<<Q;
	if(vflag >= 1) cout<<cmd.str()<<endl;
	string result=get_std_from_cmd(cmd.str().c_str());
	if(vflag >= 1) cout<<result<<endl;
	vector<string> lines=strsplit(result,"\n");
	vector<int> myconf;
	for(vector <string>::iterator line=lines.begin();line!=lines.end();++line){
		vector <string> mystr=strsplit(*line," ");
		for(vector <string>::iterator it=mystr.begin();it!=mystr.end();++it){
			myconf.push_back(atoi(it->c_str()));
		}
	}
	assert(myconf.size() == N);
	conf_infer.resize(N);
	for(int i=0;i<N;i++) conf_infer[i]=myconf[i];
	cout<<"spectral clustering: fraction="<<compute_config_ovl()<<endl;
}
//}}}
//{{{vector <string> strsplit(const string& str, const string& delimiters = " ")
vector <string> strsplit(const string& str, const string& delimiters = " ")
{
	vector <string> tokens;
	// Skip delimiters at beginning.
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos){
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
	return tokens;
}
//}}}

