/**
 * @Author: Sadamori Kojaku
 * @Date:   2022-11-02 13:02:40
 * @Last Modified by:   Sadamori Kojaku
 * @Last Modified time: 2022-11-28 14:58:20
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
//{{{header files
#include "bm.h"
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sstream>
#include <getopt.h>
//}}}
//{{{long get_cpu_time(void)
long get_cpu_time(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ( ru.ru_utime.tv_sec*1000 +
	     ru.ru_utime.tv_usec/1000+
	     ru.ru_stime.tv_sec*1000 +
	     ru.ru_stime.tv_usec/1000 );
}
//}}}
//{{{ void read_cab_file(string cab_ifname,vector<double> &pa,vector<double> &cab, int Q)
void read_cab_from_file(string cab_ifname,vector<double> &pa,vector<double> &cab, int Q)
{
	cout<<"Reading pa,cab from file... "<<cab_ifname<<flush;
	ifstream fin(cab_ifname.c_str());
	assert(fin.good() && "I can not open file which contains pa and cab");
	string tmpstr;
	bool control_line=false;
	while(true){
		if(!control_line){//control line is read, don't override it.
			if(!(fin>>tmpstr)) break;
		}
		control_line=false;
		if(tmpstr == string("pa_begin") || tmpstr == string("#Vector_na:")){
			if(!(fin>>tmpstr)) break;
			//cout<<tmpstr<<endl;
			while(tmpstr != string("pa_end")){
				if(tmpstr[0]=='#'){ control_line=true; break; }
				if(tmpstr.find(',') != string::npos){//
					vector<string> tmppa=strsplit(tmpstr,",");
					for(unsigned int i=0;i<tmppa.size();i++) pa.push_back(atof(tmppa[i].c_str()));
				}else pa.push_back(atof(tmpstr.c_str()));
				if(!(fin>>tmpstr)) break;
			}
		}else if(tmpstr == string("cab_begin") || tmpstr==string("#Matrix_cab:")){
			fin >> tmpstr;
				//cout<<tmpstr<<endl;
			while(tmpstr != string("cab_end")){
				if(tmpstr[0]=='#'){ control_line=true; break; }
				if(tmpstr.find(',') != string::npos){//
					vector<string> tmpcab=strsplit(tmpstr,",");
					for(unsigned int i=0;i<tmpcab.size();i++) cab.push_back(atof(tmpcab[i].c_str()));
				}else cab.push_back(atof(tmpstr.c_str()));
				if(!(fin>>tmpstr)) break;
			}
		}
	}
	assert((pa.size()==Q-1 || pa.size()==Q ) &&"there is something wrong in setting pa from file");
	assert((cab.size()==Q*Q || cab.size()==Q*(Q+1)/2 ) &&"there is something wrong in setting cab from file");
	fin.close();
	cout<<"done"<<endl;
}
//}}}
	//{{{void show_help_short()
	void show_help_short(char **argv)
	{
		cout<<"Generate, inference and learning by Stochastic Block Model"<<endl;
		cout<<"Usage: "<<argv[0]<<" [gen/infer/learn] options. For more information, try "<<argv[0]<<" -h"<<endl;
		exit(0);
	}
	//}}}
//{{{void show_help()
void show_help(char **argv)
{
	string deli(" ");
	cout<<"Generate, inference and learn by Stochastic Block Model"<<endl;
	cout<<"Usage: "<<argv[0]<<" [gen/infer/learn/spc/walk] options"<<endl;
	cout<<"Options:"<<endl;
	cout<<" -d: set dc_flag, degree corrected model. Valid values are 0, 1 and 2. 0: vanilla block model. 1: degre corrected model 1. 2: degree corrected model 2. Default 0."<<endl;
	cout<<" -l: set lfname, name of Graph file"<<endl;
	cout<<" -r: set learning_rate, from 0.0 to 1.0. The larger value, the faster learning. Default 0.2."<<endl;
	cout<<" -R: set dumping_rate, from 0.0 to 1.0. The larger value, the faster converge. default 1.0(no dumping)."<<endl;
	cout<<" -e: set bp_conv_crit, convergence criterium of BP and MF, from 0.0 to 1.0. Default 5.0e-6."<<endl;
	cout<<" -E: set learning_conv_crit, convergence criterium of learning, from 0.0 to 1.0. Default 1.0e-6."<<endl;
	cout<<" -t: set time_conv, maximum time for BP to converge, default 100."<<endl;
	cout<<" -i: set init_flag, flag to initialize BP. Valid values are 0, 1 and 2. Defualt 1."<<endl;
	cout<<"     0 means do not initialize BP or MF, 1 means initialize by random messages, 2 means initialize by planted configuration."<<endl;
	cout<<" -q: set Q, number of communities, default 2."<<endl;
	cout<<" -p: set pa, use ',' to separate."<<endl;
	cout<<" -c: set cab, use ',' to separate."<<endl;
	cout<<" -P: set epsilon and c, use ',' to separate. Then pa and cab are set from epsilon and c."<<endl;
	cout<<" -D: set randseed, default value is set from current time, which changes every call."<<endl;
	cout<<" -v: set verbose flag, valid values are -1, 0, 1, 2, 3. Larger value gives more output message. Default 0."<<endl;
	cout<<" -w: set gml_fname, name of file to write graph in gml format."<<endl;
	cout<<" -W: set spm_fname, name of file to write graph in spm format (which is used by mixnet)."<<endl;
	cout<<" -L: set cab_ifname, name of file which contains pa and cab."<<endl;
	cout<<" -M: set marginal_fname, name of file that "<<argv[0]<<" will output the marginals and maxarg configuration into."<<endl;
	cout<<" --wcab: set cab_ofname, name of file that "<<argv[0]<<" will output the (learned) matrix into."<<endl;
	cout<<" --spcmode: set mode of spectral clustering, valid values are [0,1,2,3,4]. 0: use Laplacian; 1: use Lrw; 2: use Lsym; 3: use P; 4: use Newman's method with modularity matrix. Defult 3."<<endl;
	cout<<" -h: show help and exit"<<endl;
	cout<<endl<<"To generate a graph one needs to provide number of nodes using -n parameter, number of comminities using -q parameter and pa and cab matrix. The graph will be written in gml format into filename given by -w parameter or spm format give by -W parameter."<<endl;
	cout<<endl<<"There are four ways to set them vector pa and matrix cab:"<<endl;
	cout<<deli<<"1.) Use -p parameter to set pa[0],pa[1],...,pa[q-1], separated by ','. and use -c to set cab[0][1],...,cab[0][q-1],cab[1][1],...,cab[1][q-1], separated by ','."<<endl;
	cout<<deli<<"2.) Use epsilon and c to set pa and cab automatically(epsilon-c mode). In this mode, you need to provide N using -n, and epsilon,c by -P at same time, separated by ','. Example: -P 0.15,3.0 sets epsilon=0.15 and c=3.0."<<endl;
	cout<<deli<<"3.) Set pa and cab from file give by -L parameter."<<endl;
	cout<<deli<<"4.) If none of options is selected, which means no '-p -c' or '-P' or '-L' is given, "<<argv[0]<<" will set pa and cab randomly."<<endl;
	cout<<endl<<"Examples:"<<endl;
	cout<<"Generating with 1000 nodes, 2 groups, average degree 3 and epsilon=0.15. Graph is written into a.gml:"<<endl;
	cout<<deli<<argv[0]<<" gen -n1000 -q2 -P0.15,3.0 -w a.gml"<<endl;
	cout<<"Inference from graph a.gml, by BP. pa and cab matrix are set by epsilon=0.15 and c=3.0: "<<endl;
	cout<<deli<<argv[0]<<" infer -l a.gml -n1000 -q2 -P0.15,3.0 -v1"<<endl;
	cout<<"Inference from graph a.gml, by Mean-field. pa and cab matrix is set directly by -p and -c parameters: "<<endl;
	cout<<deli<<argv[0]<<" infer -l a.gml -q2 -p0.5 -c10,2,10 -v1 -m nmf"<<endl;
	cout<<"Learning with 4 communities and random initial matrix, using vanilla model: "<<endl;
	cout<<deli<<argv[0]<<" learn -q 4 -l a.gml"<<endl;
	cout<<"Learning with 4 communities and random initial matrix, using degree corrected model 1: "<<endl;
	cout<<deli<<argv[0]<<" learn -q 4 -l a.gml -d 1"<<endl;
	cout<<"Learning with 2 communities and given parameters: "<<endl;
	cout<<deli<<argv[0]<<" learn -l a.gml -n10000 -P0.10,3.0 -v0"<<endl;
	cout<<"Learning with 4 communities and given parameters: "<<endl;
	cout<<deli<<argv[0]<<" learn -q4 -l a.gml -p0.25,0.25,0.25 -c50,5,5,5,50,5,5,50,5,50 -e1.0e-8 -E 1.0e-8 -r 0.1 -R0.1 -D100"<<endl;
	cout<<"Learning with 2 communities, pa and cab are taken from file '1.cab': "<<endl;
	cout<<deli<<argv[0]<<" learn -l a.gml -n10000 -v0 -L 1.cab"<<endl;
	cout<<"Learning starting from a na,cab matrix given by spectral clustering using vanilla model:"<<endl;
	cout<<deli<<argv[0]<<" spc -l a.gml -q4 -d 0"<<endl;
	cout<<"Learning starting from a na,cab matrix given by spectral clustering using degree corrected model 1:"<<endl;
	cout<<deli<<argv[0]<<" spc -l a.gml -q4 -d 1"<<endl;
	cout<<"Learning starting from a na,cab matrix given by Newman's leading eigenvector of modularity matrix method:"<<endl;
	cout<<deli<<argv[0]<<" spc -l a.gml -q4 --spcmode 4"<<endl;
	exit(0);
}
//}}}
//{{{ void parse_command_line(int argc, char **argv, string & lfname, double &learning_rate, double &dumping_rate, double &bp_conv_crit, double &learning_conv_crit, int &time_conv, int &Q, int &randseed, int &learning_max_time,string& pastr, string& cabstr, int &infer_mode, string &eps_c_str, int &init_flag, int &N, string &afname, int &vflag, string &gml_fname, string &spm_fname)
void parse_command_line(int argc, char **argv, string & lfname, double &learning_rate, double &dumping_rate, double &bp_conv_crit, double &learning_conv_crit, int &time_conv, int &Q, int &randseed, int &learning_max_time,string& pastr, string& cabstr, int &infer_mode, string &eps_c_str, int &init_flag, int &N, string &afname, int &vflag, string &gml_fname, string &spm_fname, string &cab_ifname,string &marginal_fname,bool &no_learn_pa,int &wblock, double &wstep, string &Aij_fname, int &spcmode, int &dc_flag, int &LARGE_DEGREE,string &cab_ofname, int &genmode)
{
	static struct option long_options[] =
	{
//		{"brief",   no_argument,       &verbose_flag, 0},
		{"help",     no_argument,       0, 'h'},
		{"load",  required_argument, 0, 'l'},
		{"degree",  required_argument, 0, 'd'},
		{"n",  required_argument, 0, 'n'},
		{"verbose",    required_argument, 0, 'v'},
		{"lrate",    required_argument, 0, 'r'},
		{"drate",    required_argument, 0, 'R'},
		{"bperr",    required_argument, 0, 'e'},
		{"lerr",    required_argument, 0, 'E'},
		{"tconv",    required_argument, 0, 't'},
		{"tlearning",    required_argument, 0, 'T'},
		{"pa",    required_argument, 0, 'p'},
		{"epsc",    required_argument, 0, 'P'},
		{"cab",    required_argument, 0, 'c'},
		{"init",    required_argument, 0, 'i'},
		{"mode",    required_argument, 0, 'm'},
		{"q",    required_argument, 0, 'q'},
		{"gmlfname",    required_argument, 0, 'w'},
		{"spm",    required_argument, 0, 'W'},
		{"append",    required_argument, 0, 'A'},
		{"loadcab",    required_argument, 0, 'L'},
		{"wcab",    required_argument, 0, 21},
		{"marginal",    required_argument, 0, 'M'},
		{"seed",    required_argument, 0, 'D'},
		{"wblock",    required_argument, 0, 111},
		{"wstep",    required_argument, 0, 122},
		{"nopa",    no_argument, 0, 120},
		{"writeA",   required_argument, 0, 121},
		{"spcmode",   required_argument, 0, 123},
		{"ld",   required_argument, 0, 20},
		{"genmode",    required_argument, 0, 22},
		{0, 0, 0, 0}
	};
	/* getopt_long stores the option index here. */
	int option_index = 0;
	char cc;
	while((cc=getopt_long(argc,argv,"l:r:R:e:E:t:q:D:v:s:L:c:I:T:p:P:i:m:n:w:W:v:d:M:A:h",long_options,&option_index))!=-1){
	  switch(cc){
			case 'l':	lfname=optarg;	break;
			case 'n':	N=atoi(optarg); break;
			case 'd':	dc_flag=atoi(optarg); break;
			case 'v':	vflag=atoi(optarg); break;
			case 'r':	learning_rate=atof(optarg); break;
			case 'R':	dumping_rate=atof(optarg); break;
			case 'e':	bp_conv_crit=atof(optarg); break;
			case 'E':	learning_conv_crit=atof(optarg); break;
			case 't':	time_conv=atoi(optarg); break;
			case 'T':	learning_max_time=atoi(optarg); break;
			case 'p':	pastr=optarg; break;
			case 'P':	eps_c_str=optarg; break;
			case 'c':	cabstr=optarg; break;
			case 'i':	init_flag=atoi(optarg); break;
			case 'm':	if(strcmp(optarg,"bp") == 0) infer_mode=1;else if(strcmp(optarg,"nmf") == 0) infer_mode=0;else {show_help(argv); exit(0);} break;
			case 'q':	Q=atoi(optarg); break;
			case 'w':	gml_fname=optarg;	break;
			case 'W':	spm_fname=optarg;	break;
			case 'A':	afname=optarg; break;
			case 'L':	cab_ifname=optarg; break;
			case 21:	cab_ofname=optarg; break;
			case 'M':	marginal_fname=optarg; break;
			case 'D':	randseed=atoi(optarg); break;
			case 120:	no_learn_pa=true; break;
			case 121:	Aij_fname=optarg; break;
			case 111:	wblock=atoi(optarg); break;
			case 122:	wstep=atof(optarg); break;
			case 123:	spcmode=atoi(optarg); break;
			case 20:	LARGE_DEGREE=atoi(optarg); break;
			case 22:	genmode=atoi(optarg); break;
			//case 'd':	LARGE_DEGREE=atoi(optarg); break;
			case 'h':	show_help(argv);
		}
	}
}
//}}}
//{{{ int main(int argc, char** argv)
int main(int argc, char** argv)
{
	//read the first argument to set the function.
	if(argc==1) show_help_short(argv);
	string function;
	if(argc > 1){
		function=argv[1];
		if(function == string("-h")||function == string("--help")) show_help(argv);
		else if(function != string("gen") && function != string("infer") && function != string("learn")&& function != string("walk")&& function != string("spc")) show_help_short(argv);
	}
	double learning_rate=1.0,dumping_rate=1.0,bp_conv_crit=1.0e-3,learning_conv_crit=5.0e-8,wstep=0.02;
	long t_begin=get_cpu_time();
	int N=-1,time_conv=100,Q=2,randseed=0,learning_max_time=1000,infer_mode=1,init_flag=1,vflag=0,wblock=200,spcmode=3,LARGE_DEGREE=50,genmode=0;
	bool bm_init_random=false,no_learn_pa=false;//no_learn_pa is true means does not learn pa in the learning.
	int dc_flag=0;//degree corrected model?
	string lfname,pastr,cabstr,eps_c_str,gml_fname("tmp.gml"),afname,spm_fname,cab_ifname,cab_ofname,marginal_fname,Aij_fname;
	char **myargv;
	myargv=new char *[argc-1];
	myargv[0]=new char[1024];
	strcpy(myargv[0],argv[0]);
	for(int i=1;i<argc-1;i++){
		myargv[i]=new char[1024];
		strcpy(myargv[i],argv[i+1]);
	}
	parse_command_line(argc, argv, lfname, learning_rate, dumping_rate, bp_conv_crit, learning_conv_crit, time_conv, Q, randseed, learning_max_time, pastr, cabstr, infer_mode, eps_c_str, init_flag, N, afname, vflag,gml_fname,spm_fname,cab_ifname,marginal_fname,no_learn_pa,wblock,wstep,Aij_fname,spcmode,dc_flag,LARGE_DEGREE,cab_ofname,genmode);
	if(dc_flag>0) cout<<"DEGREE CORRECTED BLOCK MODEL, mode "<<dc_flag<<endl;
	else cout<<"VANILLA BLOCK MODEL"<<endl;
	cout<<"LARGE_DEGREE="<<LARGE_DEGREE<<endl;
	if(randseed==0) randseed=int(time(NULL));
	//{{{ set pa and cab
	vector<double> pa, cab;
	if(!cab_ifname.empty()){ // set pa and cab from file named cab_ifname.
		read_cab_from_file(cab_ifname,pa,cab,Q);
	}else if(!eps_c_str.empty()){ //epsilon,c mode. Set by -P parameter
		if(genmode == 1) {
			cout<<"core-periphery networks"<<endl;
			Q=2;
		}
		assert(genmode == 0 || genmode ==1|| genmode ==2);
		assert(N != -1 && "you need to specify N by -n parameter!");
		vector<string> tmp_eps_c=strsplit(eps_c_str,",");
		assert(tmp_eps_c.size() == 2);
		double eps=atof(tmp_eps_c[0].c_str());
		double c=atof(tmp_eps_c[1].c_str());
		assert(c>0.0);
		double cin=0,co=0;
		for(unsigned int q=0;q<Q-1;q++) pa.push_back(1.0/Q);
		if(genmode==0){
			if(eps<0 || eps>99){
				cin=0;
				co=c*Q/(Q-1);
			}else{
				double eta=1.0/Q;
				cin=c/(Q*(Q-1)*eps*eta*eta+Q*eta*(eta*(eta*N-1)/N/eta));
				co=eps*cin;
			}
			for(unsigned int q=0;q<Q;q++){
				if(dc_flag==1) cab.push_back(cin);
				else if(dc_flag==2) cab.push_back(cin/c/c);
				else cab.push_back(cin);
				for(int t=q+1;t<Q;t++) {
					if(dc_flag==1) cab.push_back(co);
					else if(dc_flag==2) cab.push_back(co/c/c);
					else cab.push_back(co);
				}
			}
		}else if(genmode==1){
			//core-periphery model that all nodes have different average degree.
			assert(Q==2);
			double eta=1.0/Q;
			cin=4.0*c/(3.0+eps);
			co=eps*cin;
			cab.push_back(cin);
			cab.push_back(cin);
			cab.push_back(co);
		}else if(genmode==2){
			//core-periphery model that all nodes have same average degree.
			assert(Q==2);
			pa[0]=0.66666667;
			pa[1]=0.33333334;
			cin=9.0*c/(8.0-eps);
			double cio=(1.0-0.5*eps)*cin;
			co=eps*cin;
			cab.push_back(cin);
			cab.push_back(cio);
			cab.push_back(co);
		}


		cout<<"eps_c mode: eps="<<eps<<" c="<<c<<" cin="<<cin<<" cout="<<co<<" dc_flag="<<dc_flag<<endl;
	}else if(!pastr.empty() && !cabstr.empty() ){// pa and cab are set by -p and -c parameter.
		vector<string> tmppa=strsplit(pastr,",");
		for(unsigned int i=0;i<tmppa.size();i++) pa.push_back(atof(tmppa[i].c_str()));
		vector<string> tmpcab=strsplit(cabstr,",");
		for(unsigned int i=0;i<tmpcab.size();i++) cab.push_back(atof(tmpcab[i].c_str()));
	}else{
		bm_init_random=true;
	}
	//}}}
	cout<<"filename="<<lfname<<" learning_rate="<<learning_rate<<" dumping_rate="<<dumping_rate<<" bp_conv_crit="<<bp_conv_crit<<" learning_conv_crit="<<learning_conv_crit<<" time_conv="<<time_conv<<" Q="<<Q<<" randseed="<<randseed<<" init_flag="<<init_flag<<" vflag="<<vflag<<endl;
	blockmodel bm;
	bm.LARGE_DEGREE=LARGE_DEGREE;
	bm.init_random_number_generator(randseed);
	bm.set_Q(Q);
	bm.set_vflag(vflag);
	if(dc_flag>0) bm.set_dc(dc_flag);

	if(function == string("gen")){ // generation
		bm.graph_gen_graph(N, pa, cab);
		bm.graph_write_gml(gml_fname.c_str());
		//bm.graph_write_csv(gml_fname.c_str())
		if(!spm_fname.empty()) bm.graph_write_spm(spm_fname.c_str());
		if(!Aij_fname.empty()) bm.graph_write_A(Aij_fname.c_str());
	}else if(function == string("infer")){ // inference
		assert(!lfname.empty());
		bm.graph_read_graph(lfname);
		bm.bm_allocate();
		bm.bm_init(pa,cab);
		bm.do_inference(infer_mode, init_flag,bp_conv_crit, time_conv, dumping_rate);
		if(!marginal_fname.empty()) bm.output_marginals(marginal_fname);
	}else if(function == string("learn")){ // learning
		assert(!lfname.empty());
		if(no_learn_pa) cout<<"learn only cab !"<<endl;
		bm.graph_read_graph(lfname);
		bm.bm_allocate();
		if(!eps_c_str.empty() && dc_flag==1){ //epsilon,c mode. Set by -P parameter
			for(vector<double>::iterator it=cab.begin();it != cab.end(); ++it){
				vector<string> tmp_eps_c=strsplit(eps_c_str,",");
				double c=atof(tmp_eps_c[1].c_str());
				*it = (*it)*bm.graph_max_degree*bm.graph_max_degree/c/c;
			}
		}
		if(bm_init_random) {
			cout<<"bm(matrices) is randomly initialized!"<<endl;
			bm.bm_init_random();
		}
		else bm.bm_init(pa,cab);
		bm.EM_learning(infer_mode,init_flag,bp_conv_crit,time_conv,dumping_rate,learning_rate, learning_max_time, learning_conv_crit,no_learn_pa);
		cout<<endl<<"Matrix learned:"<<endl;
		bm.bm_show_na_cab();
		if(!marginal_fname.empty()) bm.output_marginals(marginal_fname);
		//bm.graph_write_gml(gml_fname.c_str()); // ADD BY AURELE
		double energy=bm.compute_argmax_energy();
		bm.graph_write_csv(gml_fname.c_str(), energy); // ADD BY AURELE
	}else if(function ==string("spc")){
		assert(!lfname.empty());
		bm.graph_read_graph(lfname);
		bm.bm_allocate();
		if(spm_fname.empty()) spm_fname=lfname+string(".spm");
		if(!ifstream(spm_fname.c_str())) bm.graph_write_spm(spm_fname.c_str());
		bm.spec(spm_fname,lfname,spcmode);
		double ovl_spec=bm.compute_config_ovl();
		bm.bm_build_na_cab_from_conf();
		bm.EM_learning(infer_mode,init_flag,bp_conv_crit,time_conv,dumping_rate,learning_rate, learning_max_time, learning_conv_crit,no_learn_pa);
		bm.bm_show_na_cab();
	}else if(function == string("walk")){ // learning
		assert(!lfname.empty());
		if(no_learn_pa) cout<<"learn only cab !"<<endl;
		bm.graph_read_graph(lfname);
		bm.bm_allocate();
		if(bm_init_random) {
			cout<<"bm(matrices) is randomly initialized!"<<endl;
			bm.bm_init_random();
		}
		else bm.bm_init(pa,cab);
		bm.walk_learning(infer_mode,init_flag,bp_conv_crit,time_conv,dumping_rate,learning_max_time, wblock, wstep, no_learn_pa);
		cout<<endl<<"Matrix learned:"<<endl;
		bm.bm_show_na_cab();
		if(!marginal_fname.empty()) bm.output_marginals(marginal_fname);
	}else{
		show_help(argv);
	}
	if(function != string("gen")){
		double ovl_EM=bm.compute_overlap_fraction();
		bm.compute_conductance();
		cout<<"conductance="<<bm.graph_min_conductance<<endl;
		if(!afname.empty()){
			double ovl_true=bm.compute_overlap_fraction();
			double ovl_EM=bm.compute_overlap_marg();
			double f=bm.bp_compute_f();
			double L=bm.compute_log_likelihood(f);
			double energy=bm.compute_argmax_energy();
			ostringstream cmd;
			//cmd<<"echo "<<bm.N<<" "<<Q<<" "<<randseed<<" "<<f<<" "<<L<<" "<<ovl_EM<<" "<<bm.graph_min_conductance<<" >>"<<afname;
			if(!eps_c_str.empty()){
				assert(N != -1 && "you need to specify N by -n parameter!");
				vector<string> tmp_eps_c=strsplit(eps_c_str,",");
				assert(tmp_eps_c.size() == 2);
				double eps=atof(tmp_eps_c[0].c_str());
				double c=atof(tmp_eps_c[1].c_str());
				assert(c>0.0);
				cmd<<"echo "<<eps<<" "<<c<<" "<<bm.N<<" "<<Q<<" "<<randseed<<" "<<f<<" "<<energy<<" "<<L<<" "<<ovl_EM<<" "<<ovl_true<<" "<<bm.graph_min_conductance<<" "<<bm.bp_last_diff<<" "<<bm.bp_last_conv_time<<" >>"<<afname;
			}else{
			cmd<<"echo "<<bm.N<<" "<<Q<<" "<<randseed<<" "<<f<<" "<<L<<" "<<ovl_EM<<" "<<ovl_true<<" "<<bm.graph_min_conductance<<" "<<bm.bp_last_diff<<" "<<bm.bp_last_conv_time<<" >>"<<afname;
				}
			int rcode=system(cmd.str().c_str());
			cout<<"Result has been written into "<<afname<<endl;
		}
		cout.setf(ios_base::floatfield);
	}
	if(!cab_ofname.empty()) bm.bm_show_na_cab(cab_ofname);
	cout<<"time used: "<<(get_cpu_time()-t_begin)/1000.0<<" seconds."<<endl;
}
//}}}

