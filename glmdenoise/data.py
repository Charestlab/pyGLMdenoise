"""
GLMdenoise in python

"""

def GLMdenoisedata(design,data,stimdur,tr,hrfmodel='optimise',hrfknobs=None,opt=None,figuredir=None):

    ######################### DEAL WITH INPUTS, ETC.
    
    # default hrfmodel is optimise
    
    if hrfknobs is None:
        if hrfmodel == 'fir':
            hrfknobs = 20
        else:
            hrfknobs = normalizemax(getcanonicalhrf(stimdur,tr).T)
    
    if opt is None:
        opt = dict()
    
    if figuredir is None:
        figuredir = 'GLMdenoisefigures'
    
    # massage input
    if not isinstance(design, dict):
        design = {design}
    
    # make sure the data is in the right format
    if not isinstance(data,dict):
        data[0] = data
    
    # reduce precision to single
    for p, dataset in enumerate(data):
        if not isinstance(dataset, np.float32):
            print(f'***\n\n\n##########\n GLMdenoisedata: converting data in run {p} to single format (consider doing this before the function call to reduce memory usage). \n\n\n##########\n***\n')
            data[p] = np.single(data[p])   
    
    # do some error checking
    if any(np.isfinite(data[1])):
        print('***\n\n\n##########\n GLMdenoisedata: WARNING: we checked the first run and found some non-finite values (e.g. NaN, Inf). unexpected results may occur due to non-finite values. please fix and re-run GLMdenoisedata. \n\n\n##########\n***\n')
    