#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Tools for analysis of chronic electrophysiology and behavior data.
Hengen Lab
Washington University in St. Louis
Author: Sam Brunwasser
Email: sbrunwa@wustl.edu
Version:  0.1
'''

import glob
import numpy as np
import pandas as pd
import musclebeachtools as mbt

def bin_spikes(neurons,binsize,starttime=0,endtime=13*3600,fs=25000):

    spiketimes = np.zeros([len(neurons),np.int((endtime-starttime)/binsize)])

    for i,cell in enumerate(neurons):
        edges = np.arange(starttime,endtime+(binsize*0.9),binsize)
        counts, bins = np.histogram(cell.spike_time/fs,edges)

        spiketimes[i,:] = counts
    return spiketimes

def bin_spikes_1d(spikes,binsize,tspan):
    
    edges = np.arange(0,tspan+binsize,binsize)
    frarray, bins = np.histogram(spikes,edges)
    
    return frarray

def nrnlistinfo(nrnlist):
    '''
    Prints basic info about a list of neuron objects, including number of neurons in the list and length of recording.
    Prints information about animal, region, condition if available.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    '''
    import numpy as np
    import musclebeachtools as mbt

    neurons = np.array(nrnlist)
    nrncount = neurons.shape[0]
    print('There are ' + '\033[1m' + str(nrncount) + ' neurons \033[0m in this file.' )
    maxtimes = np.zeros(nrncount)
    for i, cell in enumerate(neurons):
        maxtimes[i] = np.max(cell.spike_time)/(cell.fs*3600)

    reclength = np.round(np.mean(maxtimes),2)
    print('Recording is \033[1m ' + str(reclength) + ' hours \033[0m long. ')

    #add print statement for number of animals in nrnlist
    if hasattr(neurons[0],'animal'):
        print('These neurons are from the following animal(s):')
        print(np.unique([n.animal for n in neurons]))

    #add print statement for number of region in nrnlist
    if hasattr(neurons[0],'region'):
        print('These neurons are in the following region(s):')
        print(np.unique([n.region for n in neurons]))

    #add print statement for number of conditions in nrnlist
    if hasattr(neurons[0],'condition'):
        print('These neurons are under the following conditions(s):')
        print(np.unique([n.condition for n in neurons]))

def nrnlist_by_quality(nrnlist,quals=[1]):
    '''
    Filter list of neuron objects to only select neurons from a defined quality.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    quals - list - [1] for only quality 1 cells, [1,2] for quality 1 and 2 cells, etc

    Returns:
    n_filtered - filtered list of mbt.neuron class objects matching input criteria

    '''
    import numpy as np
    import musclebeachtools as mbt

    if hasattr(nrnlist[0],'quality'):
        n_filtered = []
        for n in nrnlist:
            if (n.quality in quals):
                n_filtered.append(n)
        return(n_filtered)
    else:
        print("Neurons don't have quality attribute.")

def nrnlist_by_animal(nrnlist,animal):
    '''
    Filter list of neuron objects to only select neurons from a defined animal.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    animal - str - name of animal to select neurons from, e.g. 'CAF69'

    Returns:
    n_filtered - filtered list of mbt.neuron class objects matching input criteria

    '''
    import numpy as np
    import musclebeachtools as mbt

    if hasattr(nrnlist[0],'animal'):
        n_filtered = []
        for n in nrnlist:
            if n.animal == animal:
                n_filtered.append(n)
        return(n_filtered)
    else:
        print("Neurons don't have animal attribute.")

def nrnlist_by_region(nrnlist,region):
    '''
    Filter list of neuron objects to only select neurons from a defined region.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    region - str - name of region to select neurons from, e.g. 'V1'

    Returns:
    n_filtered - filtered list of mbt.neuron class objects matching input criteria

    '''
    import numpy as np
    import musclebeachtools as mbt

    if hasattr(nrnlist[0],'region'):
        n_filtered = []
        for n in nrnlist:
            if n.region == region:
                n_filtered.append(n)
        return(n_filtered)
    else:
        print("Neurons don't have region attribute.")

def nrnlist_by_condition(nrnlist,condition):
    '''
    Filter list of neuron objects to only select neurons from a defined condition.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    condition - str - name of animal to select neurons from, e.g. 'WT'

    Returns:
    n_filtered - filtered list of mbt.neuron class objects matching input criteria

    '''
    import numpy as np
    import musclebeachtools as mbt

    if hasattr(nrnlist[0],'condition'):
        n_filtered = []
        for n in nrnlist:
            if n.condition == condition:
                n_filtered.append(n)
        return(n_filtered)
    else:
        print("Neurons don't have condition attribute.")

def nrnlist_by_celltype(nrnlist,celltype):
    '''
    Filter list of neuron objects to only select neurons from a defined cell type.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    celltype - str - name of celltype, e.g. 'RSU' or 'FS'

    Returns:
    n_filtered - filtered list of mbt.neuron class objects matching input criteria

    '''
    import numpy as np
    import musclebeachtools as mbt

    if hasattr(nrnlist[0],'cell_type'):
        n_filtered = []
        for n in nrnlist:
            if n.cell_type == celltype:
                n_filtered.append(n)
        return(n_filtered)
    else:
        print("Neurons don't have condition attribute.")

def nrnlist_by_genotype(nrnlist,genotype='WT'):
    '''
    Filter list of neuron objects to only select neurons from a defined quality.

    Inputs:
    nrnlist - list of mbt.neuron class objects
    genotype - str - name of genotype desired

    Returns:
    n_filtered - filtered list of mbt.neuron class objects matching input criteria

    '''
    import numpy as np
    import musclebeachtools as mbt
    
    n_filtered = []
    
    if genotype in ['WT','wt','wildtype']:
        genos = ['WT','wt','wildtype']
    elif genotype in ['APP','app','app_ps1','APP/PS1','appps1']:
        genos = ['APP','app','app_ps1','APP/PS1','appps1']
        
    for n in nrnlist:
        if hasattr(n,'condition'):
            if n.condition in genos:
                n.genotype = genotype
                n_filtered.append(n)
        elif hasattr(n,'genotype'):
            if n.genotype in genos:
                n.genotype = genotype
                n_filtered.append(n)
        elif hasattr(n,'geno'):
            if n.geno in genos:
                n.genotype = genotype
                n_filtered.append(n)
        else:
            print(f'Neuron {n+1} is missing genotype attribute.')
    return(n_filtered)

def dtify(datetimestring):
    '''
    Usage:
    dtify('2020-12-28_11-16-26') --> returns datetime.datetime(2020, 12, 28, 11, 16, 26)
    '''
    import datetime
    try:
        dtobj = datetime.datetime.strptime(datetimestring,'%Y-%m-%d_%H-%M-%S')
    except:
        dtobj = datetime.datetime.strptime(datetimestring,'%Y-%m-%d %H:%M:%S')
    return(dtobj)

def vidfile_to_datetime(vidfile,dtified=0):
    '''
    Inputs:
    vidfile - str - e.g. 'CAF00082-20210304T125328-135328DLC_resnet50_homecageMar26shuffle1_374000.h5'
    dtified - int - default 0, if 1 will use vt.dtify to return datetime object.  Otherwise returns string.
    Returns:
    dlc_dt - str - start date and time from video file, eg '2020-12-28_23-16-33'.
    Note: if dtified = 1, dlc_dt is returned as a datetime object instead of a string
    '''
    dlcstr = vidfile.split('/')[-1].split('-')[1]
    dlcday = dlcstr.split('T')[0]
    dlctime = dlcstr.split('T')[1]
    dlcdstr = dlcday[:4] + '-' + dlcday[4:6] + '-' + dlcday[-2:]
    dlctstr = dlctime[:2] + '-' + dlctime[2:4] + '-' + dlctime[-2:]
    dlc_dt_str = dlcdstr + '_' + dlctstr
    if (dtified == 1) :
        dlc_dt = dtify(dlc_dt_str)
        return(dlc_dt)
    else:
        return(dlc_dt_str)

def bonfcorr(pvals):
    
    from statsmodels.stats.multitest import multipletests as bonferonni

    #bonferroni correction for multiple comparisons
    pvals_bonf = bonferonni(pvals)[1]
    
    return pvals_bonf

def get_signif(pvals):
    pval_labs = np.zeros(len(pvals),dtype='object')

    pval_labs[pvals > 0.05] = ''
    pval_labs[pvals < 0.05] = '*'
    pval_labs[pvals < 0.005] = '**'
    pval_labs[pvals < 0.0001] = '***'

    pval_labs = pval_labs.tolist()
    
    return pval_labs

def smooth_1darray(onedarray,binsize,sigma):
    '''
    onedarray - np.array - 1d array of binned data
    binsize - size of bins in onedarray (in seconds)
    sigma - width of gaussian kernel for smoothing (in seconds) e.g. 0.05 is a typical sigma (50 ms)
    '''

    from scipy.ndimage import gaussian_filter

    #smoothing using code copied from https://matthew-brett.github.io/teaching/smoothing_intro.html
    sig = sigma/binsize
    array_smoothed = gaussian_filter(onedarray, sigma=sig)

    return(array_smoothed)

def natural_sort(l):
    #adapted from https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def zscore_all(data):
    '''
    Perform z score on input.

    (X - mean(X)) / std(X)
    '''
    return (data - np.mean(data))/np.std(data)

def zscore_to_flank(binned, baseline_length = 20):
    '''
    Zscore input relative to a defined flank (both sides).

    Input:
    binned - np.array - 2d array of the form neurons/trials X T
    baseline_length - int - number of time points to use as baseline.

    Output:
    bin_z - input data zscored to average/std of flank period
    '''
    binbl = np.hstack([binned[:,:baseline_length],binned[:,-baseline_length:]])
    binmean = np.nanmean(binbl,axis=1)
    binstd = np.nanstd(binbl,axis=1)
    binnum = binned - np.reshape(binmean,[np.shape(binmean)[0],1])
    bin_z = binnum/np.reshape(binstd,[np.shape(binstd)[0],1])
    return(bin_z)

def zscore_from_peth(peth_mean,binsz):
    numer = np.subtract(peth_mean.T,np.mean(peth_mean[:,:int(0.3/binsz)],axis=1)).T
    denom = np.std(peth_mean[:,:int(0.3/binsz)],axis=1).reshape([peth_mean.shape[0],1])
    peth_rsu_mean_zs = numer/denom
    midrange = (int(peth_mean.shape[1]/2 - peth_mean.shape[1]/40),int(peth_mean.shape[1]/2 + peth_mean.shape[1]/40))
    peth_rsu_mean_zs = peth_rsu_mean_zs[np.argsort(-np.max(peth_rsu_mean_zs[:,midrange[0]:midrange[1]],axis=1)),:]
    return(peth_rsu_mean_zs)

def return_sleepdf(sleepfiles):
    '''
    Takes list of sleep numpy files and returns a pd dataframe of start/stop times and sleep states.
    '''
    sleep_states = np.zeros((2,0))

    for idx, f in enumerate(sleepfiles):
        sleeps = np.load(f)
        timestamps = (np.nonzero(np.diff(sleeps))[0]+1)*4
        time_ind = (timestamps/4)-1
        states = sleeps[time_ind.astype(int)]
        timestamps = timestamps+(3600*idx)
        s = np.array(states)
        t = np.stack((timestamps,s))
        sleep_states = np.concatenate((sleep_states,t), axis =1)
        last = idx

    start_time = np.append([0],sleep_states[0,:-1])
    stop_time = sleep_states[0,:]
    sleepstate = sleep_states[1,:]

    sleepdf = pd.DataFrame({'start_time': start_time,'stop_time': stop_time,'sleepstate': sleepstate})
    return(sleepdf)

def smooth_spikes(spiketimes,binsize,sigma):
    '''
    spiketimes - np.array - N neurons x T bins binned spiketimes (as returned by dimt.bin_spikes)
    binsize - size of bins passed to dimt.bin_spikes (in seconds)
    sigma - width of gaussian kernel for smoothing (in seconds) e.g. 0.05 is a typical sigma (50 ms)
    '''

    from scipy.ndimage import gaussian_filter

    spiketimes_smoothed = np.zeros(spiketimes.shape)

    ncells = spiketimes.shape[0]

    for cellno in np.arange(ncells):
        #smoothing using code copied from https://matthew-brett.github.io/teaching/smoothing_intro.html
        y_vals = spiketimes[cellno,:]
        x_vals = np.arange(y_vals.shape[0])
        sig = sigma/binsize
        smoothed_vals = gaussian_filter(y_vals, sigma=sig)
        spiketimes_smoothed[cellno,:] = smoothed_vals

    return(spiketimes_smoothed)

def get_riptimes(ripcsv, peakdist = 0.1, ampthresh = 50):
    riptimes = pd.read_csv(ripcsv)
    fivemin = 60*5
    ripplecounts = riptimes['file'].value_counts().sort_index()
    sec_elapsed = np.arange(len(ripplecounts)) * fivemin
    secelapdic = {ripplecounts.index.values[i]: sec_elapsed[i] for i in range(len(ripplecounts))}
    riptimes['sec_elapsed'] = riptimes['file'].map(secelapdic)
    riptimes = riptimes.sort_values(['file','start_time'])
    riptimes['start_time'] = riptimes['start_time'] + riptimes['sec_elapsed']
    riptimes['end_time'] = riptimes['end_time'] + riptimes['sec_elapsed']
    riptimes['peak_time'] = riptimes['start_time'] + (riptimes['length']/2)
    riptimes['ripple_number'] = np.arange(len(riptimes)) + 1
    #require that peaks be peakdist sec apart
    diff_bw_peaks = np.pad(np.diff(riptimes.peak_time),(1,0))
    riptimes = riptimes[diff_bw_peaks > peakdist]
    #amplitude criteria
    riptimes = riptimes[riptimes['amplitude']>ampthresh]
    return(riptimes)

def get_sleeptimes(sleepnpy):
    #deal with sleepybois
    sleep_states = np.zeros((2,0))
    sleeps = np.load(sleepnpy)
    timestamps = (np.nonzero(np.diff(sleeps))[0]+1)*4
    time_ind = (timestamps/4)-1
    states = sleeps[time_ind.astype(int)]
    s = np.array(states)
    t = np.stack((timestamps,s))
    sleep_states = np.concatenate((sleep_states,t), axis =1)
    start_time = np.append([0],sleep_states[0,:-1])
    stop_time = sleep_states[0,:]
    sleepstate = sleep_states[1,:]
    sleepdf = pd.DataFrame({'start_time': start_time,'stop_time': stop_time,'sleepstate': sleepstate})
    return(sleepdf)

def return_sleep_scores(f):
    sleep_states = np.zeros((2,0))
    sleeps = np.load(f)
    timestamps = (np.nonzero(np.diff(sleeps))[0]+1)*4
    time_ind = (timestamps/4)-1
    states = sleeps[time_ind.astype(int)]
    # timestamps = timestamps+(3600*idx)
    s = np.array(states)
    t = np.stack((timestamps,s))
    sleep_states = np.concatenate((sleep_states,t), axis =1)
    # last = idx
    
    start_time = np.append([0],sleep_states[0,:-1])
    stop_time = sleep_states[0,:]
    sleepstate = sleep_states[1,:]
    
    sleepstate[sleepstate==5] = 1

    sleepdf = pd.DataFrame({'start_time': start_time,'stop_time': stop_time,'sleepstate': sleepstate})
    return(sleepdf)

def riptimes_by_state(riptimes,sleepdf,state_of_interest):
    if state_of_interest == 'active':
        stateid = 1
    elif state_of_interest == 'nrem':
        stateid = 2
    elif state_of_interest == 'rem':
        stateid = 3
    elif state_of_interest == 'quiet':
        stateid = 5
    else:
        print(f'State "{state_of_interest}" not recognized')
    statestart = sleepdf[sleepdf.sleepstate==stateid].start_time
    statestop = sleepdf[sleepdf.sleepstate==stateid].stop_time
    rdfstatelist = [riptimes[(riptimes.peak_time>start) & (riptimes.peak_time < stop)] for start,stop in zip(statestart.values,statestop.values)]
    rdf_state = pd.concat(rdfstatelist)
    return(rdf_state)

def bin_and_align_spikes(neurons,eventtimes,binsize,windowsize):
    binned = np.zeros([len(neurons),int((windowsize*2)/binsize),len(eventtimes)])
    for e,event in enumerate(eventtimes):
        lbound = event - windowsize
        rbound = event + windowsize
        for c,cell in enumerate(neurons):
            spiketimes = cell.spike_time_sec
            spks = spiketimes[(spiketimes>lbound)&(spiketimes<rbound)]
            binned[c,:,e] = bin_spikes_1d(spks-lbound,binsize,windowsize*2)
    return(binned)

def get_peth_data(rippledir,wtlist,applist,state,region,celltype):
    
    wt_files = []
    for ani in wtlist:
        wt_files.append(glob.glob(f'{rippledir}*{ani}*{state.lower()}*{region.lower()}_{celltype.lower()}*mean.npy'))
    wt_files = [item for sublist in wt_files for item in sublist]
    wtdat = np.zeros([2,2000])
    for fil in wt_files:
        fildat = np.load(fil)
        wtdat = np.vstack([wtdat,fildat])
    wtdat = wtdat[2:,:]

    app_files = []
    for ani in applist:
        app_files.append(glob.glob(f'{rippledir}*{ani}*{state.lower()}*{region.lower()}_{celltype.lower()}*mean.npy'))
    app_files = [item for sublist in app_files for item in sublist]
    appdat = np.zeros([2,2000])
    for fil in app_files:
        fildat = np.load(fil)
        appdat = np.vstack([appdat,fildat])
    appdat = appdat[2:,:]
    
    return(wtdat,appdat)

def calc_isi(cell):
    time_s = cell.spike_time/cell.fs
    start = cell.start_time
    end = cell.end_time
    idx = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
    ISI = np.diff(time_s[idx])

    # plot histogram and calculate contamination
    edges = np.linspace(0, 0.1, 51)
    hist_isi = np.histogram(ISI, edges)

    # Calculate contamination percentage
    contamination = 100*(sum(hist_isi[0][0:int((0.1/0.1) *
                         (101-1)/50)])/sum(hist_isi[0]))
    contamination = round(contamination, 2)
    return(contamination)

def simulate_time(n_samples, sampling_frequency):
    # adapated from https://github.com/Eden-Kramer-Lab/ripple_detection/
    return np.arange(n_samples) / sampling_frequency

def curve_function(x, a, b, c):
    return a * np.exp(-b * x) + c

def calc_dim_from_curve(curvedat,maxval,target=0.95):
    #how many dimensions needed to reach 95% max regression performance

    components = np.arange(curvedat.shape[0])
    y = curvedat/maxval
        
    try:
        ev_cumfits, pcov = curve_fit(curve_function, components, y)
        
        if (np.sum(np.isinf(pcov)) > 0):
            dim95 = np.nan
        else:
            a_term = ev_cumfits[0]
            b_term = ev_cumfits[1]
            c_term = ev_cumfits[2]

            #derived from y = ae^(-bx)+c to solve for x
            dim95 = (-1)*(np.log((target-c_term)/a_term))/b_term

            #have to add 1 because we start with 0 index (technically 0th x position is 1 dimension)
            dim95 += 1

    except:
        dim95 = np.nan
        pcov = np.nan
    
    return(dim95)
