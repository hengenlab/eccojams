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
from matplotlib import pyplot as plt
import musclebeachtools as mbt

################    NEURON OBJECT FUNCTIONS    ################

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

def check_wfs(nrnlist):
    '''
    Display waveforms of all units in a list of neuron objects

    Inputs:
    nrnlist - list of mbt.neuron class objects

    Recommended to try ~20 neurons at a time for plotting.  Will fix this later.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import musclebeachtools as mbt

    neurons = np.array(nrnlist)
    nrncount = neurons.shape[0]

    nrows = int(np.ceil(nrncount/3))
    fig,ax = plt.subplots(nrows,3,figsize=(3*nrows,7),sharex=True)
    ax = ax.ravel()

    for i, cell in enumerate(neurons):
        ax[i].plot(cell.waveform, color='#5DADE2')
        textypos = ax[i].get_ylim()[0] + (ax[i].get_ylim()[1]-ax[i].get_ylim()[0])/5
        ax[i].text(40, textypos, str('Cell # ' + str(i)))
        
    plt.tight_layout()
    plt.show()

def check_isi(nrnlist):
    '''
    Display ISI histograms of all units in a list of neuron objects

    Inputs:
    nrnlist - list of mbt.neuron class objects

    Recommended to try ~20 neurons at a time for plotting.  Will fix this later.
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import musclebeachtools as mbt

    neurons = np.array(nrnlist)
    nrncount = neurons.shape[0]

    nrows = int(np.ceil(nrncount/3))
    fig,ax = plt.subplots(nrows,3,figsize=(3*nrows,7),sharex=True)
    ax = ax.ravel()

    for i, cell in enumerate(neurons):
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
        cont_text = str(contamination) + '% contam'
        ax[i].axvline(x=1, color='r', linestyle='dashed', linewidth=1)
        ax[i].bar(edges[1:]*1000-0.5, hist_isi[0], color='#5DADE2')
        ax[i].set_ylim(bottom=0)
        ax[i].set_xlim(left=0)
        ax[i].text(50, 0.8*ax[i].get_ylim()[1], str('Cell # ' + str(i)))
        ax[i].text(50, 0.5*ax[i].get_ylim()[1], cont_text)
        
    plt.tight_layout()
    plt.show()

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

def check_continuity(nrnlist,starttime=0,endtime=12*3600,binsize=3600,fs = 25000):
    '''
    Display firing rate heatmap of all units in a list of neuron objects

    Inputs:
    nrnlist - list of mbt.neuron class objects
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import musclebeachtools as mbt

    neurons = np.array(nrnlist)
    nrncount = neurons.shape[0]

    # starttime = 0
    # endtime = np.ceil(reclength)*3600
    # # endtime = 3*3600

    spiketimes = np.zeros([len(neurons),int((endtime-starttime)/binsize)])

    for i,cell in enumerate(neurons):
        edges = np.arange(starttime,endtime+binsize,binsize)
        counts, bins = np.histogram(cell.spike_time/fs,edges)
        spiketimes[i,:] = counts

    plt.figure()
    plt.imshow(spiketimes,cmap='gist_yarg')
    plt.xlabel('Time (hours)')
    plt.ylabel('Cell Number')
    plt.show()

    nrows = int(np.ceil(nrncount/3))
    fig,ax = plt.subplots(nrows,3,figsize=(3*nrows,7),sharex=True)
    ax = ax.ravel()


    for i in range(nrncount):
        ax[i].plot(spiketimes[i,:])
        meanfr = np.mean(spiketimes[i,:])
        stdfr = np.std(spiketimes[i,:])
        ax[i].set_ylim(meanfr-4*stdfr,meanfr+4*stdfr)
        ax[i].axhline(meanfr+2*stdfr,color='r',linestyle = '-')
        ax[i].axhline(meanfr-2*stdfr,color='r',linestyle = '-')
        ax[i].text(0, 1*ax[i].get_ylim()[1], str('Cell # ' + str(i)))

    plt.show()

def plot_raster(nrns,starttime,endtime,savefig=0,filename=''):
    # starttime = 0
    # endtime = 10

    fig, ax1 = plt.subplots(1, 1,sharex=True,gridspec_kw={'height_ratios': [len(nrns)/2]})

    cellstoplot = nrns
    spike_timestamps = []
    for cellnum in np.arange(len(cellstoplot)):
        index = (cellstoplot[cellnum].spike_time_sec>starttime)&(cellstoplot[cellnum].spike_time_sec<endtime)
        spikes = cellstoplot[cellnum].spike_time_sec[index]
        spike_timestamps.append(spikes)
    spikeraster = np.asarray(spike_timestamps)

    colors2 = [[0, 0, 0]]
    ax1.eventplot(spikeraster,colors=colors2,lineoffsets=1,linelengths=0.3,linewidths=0.5)
    # Provide the title for the spike raster plot
    ax1.set_title(filename)
    # #determine x ticks and labels
    if ((endtime-starttime)<60):
        #plot every second
        ax1.set_xlabel('Time (sec)')
        xticks_in_sec = np.arange(start = starttime,stop = endtime, step = 1)
        xlabels_in_sec = np.around(xticks_in_sec,decimals=1)
    elif ((endtime-starttime)<3600):
        #plot every minutea
        ax1.set_xlabel('Time (minutes)')
        xticks_in_min = np.arange(start = starttime,stop = endtime, step = 60)
        xlabels_in_min = np.around(xticks_in_min/60,decimals=1)
    else:
        #plot every hour
        ax1.set_xlabel('Time (hours)')
        xticks_in_hr = np.arange(start = starttime,stop = endtime, step = 3600)
        xlabels_in_hr = np.around(xticks_in_hr/3600,decimals=1)
    ax1.set_xticks(xticks_in_sec)
    ax1.set_xticklabels(xlabels_in_sec)

    # Give y axis label for the spike raster plot
    ax1.set_ylabel('Cell #')
    celllabels = [cellstoplot[cellnum].clust_idx for cellnum in np.arange(len(cellstoplot))]
    ax1.set_yticks(np.arange(len(cellstoplot)),celllabels)

    if savefig == 1:
        plt.savefig(str(filename + '.pdf'),dpi=300,bbox_inches='tight')
        plt.show()
    else:
        plt.tight_layout()
        plt.show()

def bin_spikes(neurons,binsize,starttime=0,endtime=13*3600,fs=25000):

    spiketimes = np.zeros([len(neurons),int((endtime-starttime)/binsize)])

    for i,cell in enumerate(neurons):
        edges = np.arange(starttime,endtime+(binsize*0.9),binsize)
        counts, bins = np.histogram(cell.spike_time/fs,edges)

        spiketimes[i,:] = counts
    return spiketimes

def bin_spikes_1d(spikes,binsize,tspan):
    
    edges = np.arange(0,tspan+binsize,binsize)
    frarray, bins = np.histogram(spikes,edges)
    
    return frarray

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


################    NEURON X TIME ARRAY FUNCTIONS    ################

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

def shuffle_binned_data(binned_data,randmethod = 3,plotdata=0):
    '''
    ####METHODS########
    # 1 - random shuffled
    # 2 - random shuffled, boostrapped
    # 3 - random poisson FR (statistically matched)
    # 4 - random poisson FR (statistically matched), boostrapped
    # 5 - random shift
    # 6 - random shift, bootstrapped
    '''

    #Generate dummy data from real data
    n_nrn = np.shape(binned_data)[0]
    n_fr_bins = np.shape(binned_data)[1]
    dummy_fr_list = []

    if randmethod == 1:
        ###RANDOM SHUFFLE  - no bootstrapping###
        dummy_fr = np.zeros((n_nrn,n_fr_bins))
        for i in range(n_nrn):
            mask = np.random.randint(0,n_fr_bins,n_fr_bins)
            dummy_fr[i,:] = binned_data[0,mask]
        dummy_fr_bootstrapped = dummy_fr

    elif randmethod == 2:
        ###RANDOM SHUFFLE - bootstrapped###
        for strap in range(1000):
            dummy_fr = np.zeros((n_nrn,n_fr_bins))
            for i in range(n_nrn):
                mask = np.random.randint(0,n_fr_bins,n_fr_bins)
                dummy_fr[i,:] = binned_data[0,mask]
            dummy_fr_list.append(dummy_fr)
        dummy_fr_bootstrapped = np.mean(dummy_fr_list,axis=0)

    elif randmethod == 3:
        ###POISSON METHOD - no bootstrapping###
        dummy_fr = np.zeros((n_nrn,n_fr_bins))
        for i in range(n_nrn):
            mean_fr = np.mean(binned_data[i,:])
            dummy_fr[i,:] = np.random.poisson(mean_fr,n_fr_bins)
        dummy_fr_bootstrapped = dummy_fr

    elif randmethod == 4:
        ###POISSON METHOD - bootstrapped###
        for strap in range(1000):
            dummy_fr = np.zeros((n_nrn,n_fr_bins))
            for i in range(n_nrn):
                mean_fr = np.mean(binned_data[i,:])
                dummy_fr[i,:] = np.random.poisson(mean_fr,n_fr_bins)
            dummy_fr_list.append(dummy_fr)
        dummy_fr_bootstrapped = np.mean(dummy_fr_list,axis=0)

    elif randmethod == 5:
        #random shift method
        dummy_fr = np.zeros((n_nrn,n_fr_bins))
        for i in range(n_nrn):
            randshift = np.random.randint(n_fr_bins)
            dummy_fr[i,:] = np.hstack([binned_data[i,randshift:],binned_data[i,:randshift]])
        dummy_fr_bootstrapped = dummy_fr

    elif randmethod == 6:
        #random shift method, boostrapped
        for strap in range(1000):
            dummy_fr = np.zeros((n_nrn,n_fr_bins))
            for i in range(n_nrn):
                randshift = np.random.randint(n_fr_bins)
                dummy_fr[i,:] = np.hstack([binned_data[i,randshift:],binned_data[i,:randshift]])
            dummy_fr_list.append(dummy_fr)
        dummy_fr_bootstrapped = np.mean(dummy_fr_list,axis=0)


    if plotdata==1:
        # plt.ion()
        # plt.figure(1)
        for i in range(n_nrn):
            cellno=i
            real_fr_counts, real_fr_edges = np.histogram(binned_data[cellno,:],bins=1000)
            # dummy_fr_counts, dummy_fr_edges = np.histogram(dummy_fr[cellno,:],bins=1000)
            dummy_fr_counts, dummy_fr_edges = np.histogram(dummy_fr_bootstrapped[cellno,:],bins=1000)
            # dummy_fr_counts, dummy_fr_edges = np.histogram(dummy_fr_list[1][cellno,:],bins=1000)
            fig,ax = plt.subplots()
            # plt.clf()
            ax.plot(real_fr_edges[1:],real_fr_counts,c='b')
            ax.plot(dummy_fr_edges[1:],dummy_fr_counts,c='r')
            ax.set_title('Fano Factor = ' + str(np.var(binned_data[cellno,:])/np.mean(binned_data[cellno,:])))
            plt.show()

    return dummy_fr_bootstrapped

def spiketrain_from_probability(prob_of_spike):
    '''
    Inputs: 1d array representing firing rate over time, e.g. np.sin(np.arange(tspan)) 
    Outputs: an array of spiketimes generated with poisson statistics and corrected for 1ms refractory period
    '''
    if np.sum(prob_of_spike<0) > 0:
        prob_of_spike = prob_of_spike + np.abs(np.min(prob_of_spike)) #move probability into positive space
    poisson_noise = np.random.poisson(lam=1,size=np.shape(prob_of_spike))
    # poisson_noise = np.random.randint(0,2,size=100)
    
    z = prob_of_spike * poisson_noise
    
    if prob_of_spike.ndim == 1:
        threshold = np.mean(z) + np.std(z)
    else:
        threshold = np.mean(z,axis=1) + np.std(z,axis=1)
    spikes = np.where(np.greater(z.T,threshold), 1, 0)
        
    spiketimes = np.where(spikes==1)[0]
    
    spiketimes = spiketimes[1:][np.diff(spiketimes)>1]  #correct for 1ms refractory period
    
    return(spiketimes)

def spiketrain_from_firingrate_1d(frarray, binsize):
    '''
    Inputs: 1d array representing firing rate over time 
    Outputs: an array of spiketimes generated with poisson statistics and corrected for 1ms refractory period
    '''
    listofspikes = np.array([np.sort(np.random.uniform(0,binsize,int(x))) for x in frarray])
    spiketimes = np.concatenate(listofspikes + np.arange(len(frarray))) * binsize
    return(spiketimes)

def generate_random_spikes(nspikes, tspan = 15, poisson_prob=1):
    '''
    nspikes - int - number of spikes to generate
    tspan - int - length of time series (in seconds)
    poisson_prob - 0,1 - if 1, will only include spikes passing poisson threshold
    '''
    randspikes = np.sort(np.random.uniform(0,tspan,nspikes))
    if poisson_prob==1:
        #add poisson probability
        prob_of_spike = np.random.poisson(size=nspikes)
        randspikes = randspikes[prob_of_spike>0]
    return(randspikes)

def shuffle_spikes(spiketrain, jitter_size, bootstraps=1):
    '''
    spiketrain - np.array - array of spiketimes in s
    jitter_size - int - size of jitter in ms
    bootstraps - int - number of times to shuffle
    '''
    jitter_sec = jitter_size/1e3
    jitter_boot = np.random.uniform(-jitter_sec,jitter_sec,[bootstraps,np.shape(spiketrain)[0]])
    jitter = np.sum(jitter_boot,axis=0)
    shuffled = spiketrain + jitter
    return(shuffled)


################    VIDEO/SLEEP/EVENT FUNCTIONS    ################

def load_txt_as_list(textfile):
    file1 = open(textfile, 'r')
    lines = file1.readlines()
    filelist = [line.strip('\n') for line in lines]
    return(filelist)

def natural_sort(l):
    #adapted from https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def simulate_time(n_samples, sampling_frequency):
    # adapated from https://github.com/Eden-Kramer-Lab/ripple_detection/
    return np.arange(n_samples) / sampling_frequency

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

def binfile_to_timestamp(binfile):
    timestamp = '_'.join(binfile.split('/')[-1].strip('.bin').split('_')[-2:])
    return(timestamp)

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

def eventtimes_by_state(eventtimes, sleepdf, state_of_interest):
    '''
    Inputs:
    eventtimes - np.array, list, or pd.Series - 1d list/array of event times to be split by state
    sleepdf - pd.DataFrame - dataframe of sleep states (as returned by eco.return_sleep_scores)
    state_of_interest - str - name of state to split eventtimes by e.g. 'nrem'

    Outputs:
    events_state - pd.Series - contains event times occurring within state of interest only

    '''

    eventtimes = pd.Series(eventtimes)
    state_of_interest = state_of_interest.lower()

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
    eventstatelist = [eventtimes[(eventtimes>start) & (eventtimes < stop)] for start,stop in zip(statestart.values,statestop.values)]
    events_state = pd.concat(eventstatelist)

    return(events_state)


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


################    STATISTICAL FUNCTIONS    ################

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


################    NEURAL POPULATION ANALYSES    ################

def ccg_pair(nrn1,nrn2,dt=1e-3,tspan=0.05,nsegs=3000,return_plot=True,return_ccg=False):
    '''
    Display crosscorrelogram between two units.

    Inputs:
    nrn1 - mbt.neuron class object
    nrn2 - mbt.neuron class object
    dt - bin window in s, default is 1e-3 s (1 ms)
    tspan - +/- time span to calculate CCG over in seconds, default is 0.05 s (50 ms)
    nsegs - number of spikes to reference, default is None in Keith's code - I found 3000 to work well so set that here as default
    '''
    stimes1 = nrn1.spike_time_sec
    stimes2 = nrn2.spike_time_sec

    #     print('Computing cross correlation between cells {:d} and {:d}.'.format(self.clust_idx,friend.clust_idx))
    #     print('  Parameters:\n\tTime span: {} ms\n\tBin step: {:.2f} ms'.format(int(tspan*1000),dt*1000))
    #     # select spike timestamps within on/off times for self cell
    #     stimes1 = self.spike_time_sec
    #     # select spike timestamps within on/off times for self cell
    #     stimes2 = friend.spike_time_sec
    # start timer for benchmarking
    #     t_start = time.time()
    # remove spikes at the edges (where edges are tspan/2)
    subset1      = [ (stimes1 > stimes1[0]+tspan/2) & (stimes1 < stimes1[-1] - tspan/2) ]
    subset2      = [ (stimes2 > stimes2[0]+tspan/2) & (stimes2 < stimes2[-1] - tspan/2) ]
    # line above returns an array of booleans. Convert to indices
    subset1      = np.where(subset1)[1]
    subset2      = np.where(subset2)[1]
    # check to see if nsegs is user provided or default
    if nsegs is None:
        nsegs1  = int(np.ceil(np.max(stimes1/120)))
        nsegs2  = int(np.ceil(np.max(stimes2/120)))
        nsegs   = max(nsegs1,nsegs2)
    print('\tUsing {:d} segments.'.format(nsegs))
    # Take a subset of indices. We want "nsegs" elements, evenly spaced. "segindx" therefore contains nsegs spike indices, evenly spaced between the first and last one in subset
    segindx1     = np.ceil(np.linspace(subset1[0], subset1[-1], nsegs))
    segindx2     = np.ceil(np.linspace(subset2[0], subset2[-1], nsegs))

    # The spikes pointed at by the indices in "segindx" are our reference spikes for autocorrelation calculation

    # initialize timebins
    timebins = np.arange(0, tspan+dt,   dt)
    # number of bins
    nbins    = timebins.shape[0] - 1

    XCorrs = np.zeros((nsegs,2*nbins-1),float)

    stimes2_ms = stimes2 * 1e3
    dt_ms = dt * 1e3

    # ref is the index of the reference spike
    for i, ref in enumerate(segindx1):
        ref = int(ref)
        # "t" is the timestamp of reference spike
        t = stimes1[ref]
        starttime = np.round((t - tspan) * 1e3)
        endtime = np.round((t + tspan) * 1e3)

        stimes2_tspan = stimes2_ms[(stimes2_ms>starttime)&(stimes2_ms<endtime)] #stimes within tspan of reference spike
        edges2 = np.arange(starttime+dt_ms,endtime+dt_ms,dt_ms)
        counts2, bins2 = np.histogram(stimes2_tspan,edges2)
        XCorrs[i,:] = counts2

    Y = np.mean(XCorrs,axis=0)

    if return_plot==True:

        fig = plt.figure(facecolor='white')
        fig.suptitle('Cross-correlation')

        ax         = fig.add_subplot(111, frame_on=False)
        #     ax.bar( 1000*np.arange(-tspan+dt,tspan,dt), Y, width =  0.5, color = 'k' )
        ax.plot( 1000*np.arange(-tspan+dt,tspan,dt), Y, color = 'k' )
        xlim       = int(tspan*1000) # milliseconds - set to tspan
        tickstep   = int(xlim/5) # milliseconds
        ax_ticks   = [i for i in range(-xlim,xlim+1,tickstep)]
        ax_labels  = [str(i) for i in ax_ticks]
        ax.set_xticks(ax_ticks)
        ax.set_xticklabels(ax_labels)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Lag (ms)')
        plt.show()

    if return_ccg == True:
        return(Y)

def ccg_tseries(stimes1,stimes2,dt=1e-3,tspan=0.05,nsegs=3000,return_plot=True,return_ccg=False):
    '''
    Display crosscorrelogram between two units.

    Inputs:
    stimes1 - series of discrete events, in sec (e.g. output of nrn1.spike_time_sec)
    stimes2 - series of discrete events, in sec (e.g. output of nrn2.spike_time_sec)
    dt - bin window in s, default is 1e-3 s (1 ms)
    tspan - +/- time span to calculate CCG over in seconds, default is 0.05 s (50 ms)
    nsegs - number of spikes to reference, default is None in Keith's code - I found 3000 to work well so set that here as default
    '''

    #     print('Computing cross correlation between cells {:d} and {:d}.'.format(self.clust_idx,friend.clust_idx))
    #     print('  Parameters:\n\tTime span: {} ms\n\tBin step: {:.2f} ms'.format(int(tspan*1000),dt*1000))
    #     # select spike timestamps within on/off times for self cell
    #     stimes1 = self.spike_time_sec
    #     # select spike timestamps within on/off times for self cell
    #     stimes2 = friend.spike_time_sec
    # start timer for benchmarking
    #     t_start = time.time()
    # remove spikes at the edges (where edges are tspan/2)
    subset1      = [ (stimes1 > stimes1[0]+tspan/2) & (stimes1 < stimes1[-1] - tspan/2) ]
    subset2      = [ (stimes2 > stimes2[0]+tspan/2) & (stimes2 < stimes2[-1] - tspan/2) ]
    # line above returns an array of booleans. Convert to indices
    subset1      = np.where(subset1)[1]
    subset2      = np.where(subset2)[1]
    # check to see if nsegs is user provided or default
    if nsegs is None:
        nsegs1  = int(np.ceil(np.max(stimes1/120)))
        nsegs2  = int(np.ceil(np.max(stimes2/120)))
        nsegs   = max(nsegs1,nsegs2)
    print('\tUsing {:d} segments.'.format(nsegs))
    # Take a subset of indices. We want "nsegs" elements, evenly spaced. "segindx" therefore contains nsegs spike indices, evenly spaced between the first and last one in subset
    segindx1     = np.ceil(np.linspace(subset1[0], subset1[-1], nsegs))
    segindx2     = np.ceil(np.linspace(subset2[0], subset2[-1], nsegs))

    # The spikes pointed at by the indices in "segindx" are our reference spikes for autocorrelation calculation

    # initialize timebins
    timebins = np.arange(0, tspan+dt,   dt)
    # number of bins
    nbins    = timebins.shape[0] - 1

    XCorrs = np.zeros((nsegs,2*nbins-1),float)

    stimes2_ms = stimes2 * 1e3
    dt_ms = dt * 1e3

    # ref is the index of the reference spike
    for i, ref in enumerate(segindx1):
        ref = int(ref)
        # "t" is the timestamp of reference spike
        t = stimes1[ref]
        starttime = np.round((t - tspan) * 1e3)
        endtime = np.round((t + tspan) * 1e3)

        stimes2_tspan = stimes2_ms[(stimes2_ms>starttime)&(stimes2_ms<endtime)] #stimes within tspan of reference spike
        edges2 = np.arange(starttime+dt_ms,endtime+dt_ms,dt_ms)
        counts2, bins2 = np.histogram(stimes2_tspan,edges2)
        XCorrs[i,:] = counts2

    Y = np.mean(XCorrs,axis=0)

    if return_plot==True:

        fig = plt.figure(facecolor='white')
        fig.suptitle('Cross-correlation')

        ax         = fig.add_subplot(111, frame_on=False)
        #     ax.bar( 1000*np.arange(-tspan+dt,tspan,dt), Y, width =  0.5, color = 'k' )
        ax.plot( 1000*np.arange(-tspan+dt,tspan,dt), Y, color = 'k' )
        xlim       = int(tspan*1000) # milliseconds - set to tspan
        tickstep   = int(xlim/5) # milliseconds
        ax_ticks   = [i for i in range(-xlim,xlim+1,tickstep)]
        ax_labels  = [str(i) for i in ax_ticks]
        ax.set_xticks(ax_ticks)
        ax.set_xticklabels(ax_labels)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Lag (ms)')
        plt.show()

    if return_ccg == True:
        return(Y)

def lagged_pwc(X,Y,tspan):
    # Performs pairwise correlation over +/-tspan lags
    # X - time series 1
    # Y - time series 2
    # tspan - timespan to compute PWC over (in units of X and Y bins)
    
    shiftvals = np.arange(-tspan,tspan+1)
    pwc = np.zeros(len(shiftvals))

    for i,shiftval in enumerate(shiftvals):

        #timeshift
        Y_shift = np.hstack([Y[shiftval:],Y[:shiftval]])

        #perform correlation
        pwc[i] = np.corrcoef(X,Y_shift)[0,1]

    return(pwc)

def cca_on_data(X,Y):
    '''
    Standardizes X and Y data, returns first CC of X and Y
    '''
    from sklearn.cross_decomposition import CCA

    X_std = (X-X.mean())/(X.std())
    Y_std = (Y-Y.mean())/(Y.std())

    X_std = X_std.T
    Y_std = Y_std.T

    cca = CCA()
    cca.fit(X_std, Y_std)
    X_c, Y_c = cca.transform(X_std, Y_std)

    return(X_c,Y_c)

def pca_on_data(data,scaling=0, whiten=0):
    '''
    Performs PCA on N x T matrix
    Returns PCA-transformed data and array of explained variance ratio
    '''
    from sklearn.decomposition import PCA

    if scaling==0:

        if whiten == 1:
            pca = PCA(whiten=True)
        else:
            pca = PCA()
        # pca = PCA()
        pca.fit(data.T)
        data_transformed = pca.transform(data.T)
        explvar = pca.explained_variance_ratio_
        return data_transformed, explvar

    elif scaling==1:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(data.T)
        data_scaled = scaler.transform(data.T)

        pca = PCA()
        pca.fit(data_scaled)
        data_transformed = pca.transform(data_scaled)
        explvar = pca.explained_variance_ratio_
        return data_transformed, explvar

def factoranalysis_on_data(data,scaling=0):
    '''
    Performs Factor Analysis on N x T matrix
    Returns FA-transformed data and array of explained variance ratio
    '''
    from sklearn.decomposition import FactorAnalysis

    if scaling==0:

        fa = FactorAnalysis()
        fa.fit(data.T)
        data_transformed = fa.transform(data.T)
        m = fa.components_
        n = fa.noise_variance_
        m1 = m**2
        m2 = np.sum(m1,axis=1)
        explvar = m2/np.sum(m2)
        return data_transformed, explvar

    elif scaling==1:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(data.T)
        data_scaled = scaler.transform(data.T)

        fa = FactorAnalysis()
        fa.fit(data_scaled)
        data_transformed = fa.transform(data_scaled)
        m = fa.components_
        n = fa.noise_variance_
        m1 = m**2
        m2 = np.sum(m1,axis=1)
        explvar = m2/np.sum(m2)
        return data_transformed, explvar

def isomap_on_data(data,scaling=0,n_comp = 10):
    '''
    Performs isomap on N x T matrix
    Returns isomap-transformed data
    '''
    from sklearn.manifold import Isomap

    if scaling==0:

        embedding = Isomap(n_components=n_comp)
        # pca = PCA()
        embedding.fit(data.T)
        data_transformed = embedding.transform(data.T)
        return data_transformed

    elif scaling==1:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(data.T)
        data_scaled = scaler.transform(data.T)

        embedding = Isomap(n_components=n_comp)
        embedding.fit(data_scaled)
        data_transformed = embedding.transform(data_scaled)
        return data_transformed

def curve_function(x, a, b, c):
    return a * np.exp(-b * x) + c

def calc_dim_from_curve(curvedata,maxval,target=0.95):
    #how many dimensions needed to reach 95% max regression performance
    
    from scipy.optimize import curve_fit

    components = np.arange(curvedata.shape[0])
    y = curvedata/maxval
        
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








