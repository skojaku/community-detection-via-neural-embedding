def make_filename(prefix, ext, names):
    retval = prefix
    for key in names:
        retval += "_" + str(key) + "={" + str(key) + "}"
    return retval + "." + ext


def to_paramspace(dict_list):
    if isinstance(dict_list, list) is False:
        dict_list = [dict_list]
    my_dict = {}
    cols = []
    for dic in dict_list:
        my_dict.update(dic)
        cols += list(dic.keys())
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    df = pd.DataFrame(permutations_dicts)
    df = df[cols]
    return Paramspace(df, filename_params="*")


