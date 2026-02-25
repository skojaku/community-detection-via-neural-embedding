fig_perf_vs_mixing_paramspace = to_paramspace(fig_params_perf_vs_mixing)
FIG_PERFORMANCE_VS_MIXING = j(
    FIG_DIR,
    "perf_vs_mixing",
    f"fig_{fig_perf_vs_mixing_paramspace.wildcard_pattern}.pdf",
)

# ================================
# Networks and communities
# ================================

# Convert to a paramspace
net_paramspace = to_paramspace(net_params)
NET_FILE = j(NET_DIR, f"net_{net_paramspace.wildcard_pattern}.npz")
NODE_FILE = j(NET_DIR, f"node_{net_paramspace.wildcard_pattern}.npz")

# =================
# Embedding
# =================

emb_paramspace = to_paramspace([net_params, emb_params])
EMB_FILE = j(EMB_DIR, f"{emb_paramspace.wildcard_pattern}.npz")

# ===================
# Community detection
# ===================
com_detect_paramspace = to_paramspace([net_params, com_detect_params])

# Community detection
COM_DETECT_FILE = j(COM_DIR, f"{com_detect_paramspace.wildcard_pattern}.npz")

# Community detection by clustering to embedding
com_detect_emb_paramspace = to_paramspace([net_params, emb_params, clustering_params])
COM_DETECT_EMB_FILE = j(
    COM_DIR, f"clus_{com_detect_emb_paramspace.wildcard_pattern}.npz"
)


# ==========
# Evaluation
# ==========
EVAL_EMB_FILE = j(
    EVA_DIR, f"score_clus_{com_detect_emb_paramspace.wildcard_pattern}.npz"
)
EVAL_FILE = j(EVA_DIR, f"score_{com_detect_paramspace.wildcard_pattern}.npz")

#
# Loss landscape
#
LOSS_LANDSCAPE_MODEL_LIST = ["modularity", "laplacian"]
FIG_LOSS_LANDSCAPE = j("figs", "loss_landscape", "loss_landscape_model~{model}.pdf")


# ======
# RULES
# ======
#
# network generation
#
rule generate_net_multi_partition_model:
    params:
        parameters=net_paramspace.instance,
    output:
        output_file=NET_FILE,
        output_node_file=NODE_FILE,
    wildcard_constraints:
        data="multi_partition_model",
    resources:
        mem="12G",
        time="04:00:00",
    script:
        "../scripts/generate_net_by_multi_partition_model.py"


#
# Embedding
#
rule embedding_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EMB_FILE,
    params:
        parameters=emb_paramspace.instance,
    script:
        "../scripts/embedding.py"


#
# Clustering
#
rule voronoi_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="voronoi",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "../scripts/voronoi_clustering.py"


rule kmeans_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="kmeans",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "../scripts/kmeans_clustering.py"


rule silhouette_clustering_multi_partition_model:
    input:
        emb_file=EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_EMB_FILE,
    params:
        parameters=com_detect_emb_paramspace.instance,
    wildcard_constraints:
        clustering="silhouette",
    resources:
        mem="12G",
        time="01:00:00",
    script:
        "../scripts/silhouette_kmeans.py"


rule community_detection_multi_partition_model:
    input:
        net_file=NET_FILE,
        com_file=NODE_FILE,
    output:
        output_file=COM_DETECT_FILE,
    params:
        parameters=com_detect_paramspace.instance,
    script:
        "../scripts/detect_community.py"


#
# Evaluation
#
rule evaluate_communities:
    input:
        detected_group_file=COM_DETECT_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_FILE,
    resources:
        mem="12G",
        time="00:10:00",
    script:
        "../scripts/eval_com_detect_score.py"


rule evaluate_communities_for_embedding:
    input:
        detected_group_file=COM_DETECT_EMB_FILE,
        com_file=NODE_FILE,
    output:
        output_file=EVAL_EMB_FILE,
    resources:
        mem="12G",
        time="00:20:00",
    script:
        "../scripts/eval_com_detect_score.py"


rule concatenate_results_multipartition:
    input:
        input_files=expand(
            EVAL_FILE,
            data="multi_partition_model",
            **net_params,
            **com_detect_params,
        )
        + expand(
            EVAL_EMB_FILE,
            data="multi_partition_model",
            **net_params,
            **emb_params,
            **clustering_params,
        ),
    output:
        output_file=EVAL_CONCAT_FILE,
    params:
        to_int=["n", "K", "dim", "sample", "length", "dim", "cave"],
        to_float=["mu"],
    wildcard_constraints:
        data="multi_partition_model",
    resources:
        mem="4G",
        time="00:50:00",
    script:
        "../scripts/concatenate_results.py"


#
# Plot
#
rule plot_performance_vs_mixing:
    input:
        input_file=EVAL_CONCAT_FILE,
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING,
    params:
        parameters=fig_perf_vs_mixing_paramspace.instance,
        dimThreshold=False,
        normalize=False,
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
        ],
        with_legend=(
            lambda wildcards: "True" if str(wildcards.cave) == "5" else "False"
        ),
    resources:
        mem="4G",
        time="00:50:00",
    script:
        "../scripts/plot_mixing_vs_performance.py"


rule plot_performance_vs_mixing_all:
    input:
        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing),
    output:
        output_file=FIG_PERFORMANCE_VS_MIXING_ALL.format(data="multi_partition_model"),
    run:
        shell("python workflow/scripts/pdf_nup.py {input} --nup 3x4 --outfile {output}")
