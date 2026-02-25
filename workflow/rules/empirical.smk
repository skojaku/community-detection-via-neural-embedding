# ================================
# Empirical networks
# ================================

EMPIRICAL_NETWORKS = [
    "polblog",
    "airport",
    "football",
    "polbooks",
    "cora",
    "highschool",
]

emp_net_params = {"netdata": EMPIRICAL_NETWORKS}
emp_net_paramspace = to_paramspace(emp_net_params)

EMP_NET_FILE = j(NET_DIR, f"net_{emp_net_paramspace.wildcard_pattern}.npz")
EMP_NODE_FILE = j(NET_DIR, f"node_{emp_net_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================
emp_emb_paramspace = to_paramspace([emp_net_params, emb_params])
EMP_EMB_FILE = j(EMB_DIR, f"{emp_emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
emp_com_detect_paramspace = to_paramspace([emp_net_params, com_detect_params])
EMP_COM_DETECT_FILE = j(COM_DIR, f"{emp_com_detect_paramspace.wildcard_pattern}.npz")

emp_com_detect_emb_paramspace = to_paramspace(
    [emp_net_params, emb_params, clustering_params]
)
EMP_COM_DETECT_EMB_FILE = j(
    COM_DIR, f"clus_{emp_com_detect_emb_paramspace.wildcard_pattern}.npz"
)

# ==========
# Evaluation
# ==========
EMP_EVAL_EMB_FILE = j(
    EVA_DIR, f"score_clus_{emp_com_detect_emb_paramspace.wildcard_pattern}.npz"
)
EMP_EVAL_FILE = j(EVA_DIR, f"score_{emp_com_detect_paramspace.wildcard_pattern}.npz")

# =========
# Figures
# =========
FIG_EMP_PERFORMANCE = j(FIG_DIR, "performance_empirical_{clustering}.pdf")


# ======
# RULES
# ======

rule generate_empirical_net:
    params:
        parameters=emp_net_paramspace.instance,
    output:
        output_file=EMP_NET_FILE,
        output_node_file=EMP_NODE_FILE,
    wildcard_constraints:
        data="empirical",
    resources:
        mem="4G",
        time="00:30:00",
    script:
        "../scripts/generate_empirical_network.py"


use rule embedding_multi_partition_model as embedding_empirical with:
    input:
        net_file=EMP_NET_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_EMB_FILE,
    params:
        parameters=emp_emb_paramspace.instance,


use rule voronoi_clustering_multi_partition_model as voronoi_clustering_empirical with:
    input:
        emb_file=EMP_EMB_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_COM_DETECT_EMB_FILE,
    params:
        parameters=emp_com_detect_emb_paramspace.instance,


use rule kmeans_clustering_multi_partition_model as kmeans_clustering_empirical with:
    input:
        emb_file=EMP_EMB_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_COM_DETECT_EMB_FILE,
    params:
        parameters=emp_com_detect_emb_paramspace.instance,


use rule silhouette_clustering_multi_partition_model as silhouette_clustering_empirical with:
    input:
        emb_file=EMP_EMB_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_COM_DETECT_EMB_FILE,
    params:
        parameters=emp_com_detect_emb_paramspace.instance,


use rule community_detection_multi_partition_model as community_detection_empirical with:
    input:
        net_file=EMP_NET_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_COM_DETECT_FILE,
    params:
        parameters=emp_com_detect_paramspace.instance,


use rule evaluate_communities as evaluate_communities_empirical with:
    input:
        detected_group_file=EMP_COM_DETECT_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_EVAL_FILE,


use rule evaluate_communities_for_embedding as evaluate_communities_for_embedding_empirical with:
    input:
        detected_group_file=EMP_COM_DETECT_EMB_FILE,
        com_file=EMP_NODE_FILE,
    output:
        output_file=EMP_EVAL_EMB_FILE,


rule concatenate_results_empirical:
    input:
        input_files=expand(
            EMP_EVAL_FILE,
            data="empirical",
            **emp_net_params,
            **com_detect_params,
        )
        + expand(
            EMP_EVAL_EMB_FILE,
            data="empirical",
            **emp_net_params,
            **emb_params,
            **clustering_params,
        ),
    output:
        output_file=EVAL_CONCAT_FILE,
    wildcard_constraints:
        data="empirical",
    params:
        to_int=["dim", "sample", "length"],
        to_float=[],
    resources:
        mem="4G",
        time="00:30:00",
    script:
        "../scripts/concatenate_results.py"


rule plot_empirical_performance:
    input:
        input_file=EVAL_CONCAT_FILE,
    output:
        output_file=FIG_EMP_PERFORMANCE,
    params:
        model_names=[
            "node2vec",
            "deepwalk",
            "line",
            "adjspec",
            "modspec",
            "leigenmap",
            "nonbacktracking",
            "bp",
            "flatsbm",
            "infomap",
        ],
    resources:
        mem="4G",
        time="00:30:00",
    script:
        "../scripts/plot_performance_empirical_net.py"


rule plot_empirical_performance_all:
    input:
        expand(FIG_EMP_PERFORMANCE, data="empirical", clustering=clustering_params["clustering"]),
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING_ALL.format(data="empirical"),
    run:
        shell("python workflow/scripts/pdf_nup.py {input} --nup 1x3 --outfile {output}")
