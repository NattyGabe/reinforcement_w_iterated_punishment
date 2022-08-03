#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numba
import math




def gen_sig_rns(n, trms, rein1, rein2, pun1, pun2, iters, pls, rgs):
    rng = rgs[n]
    
    
    
    # number of urns for each signaler
    numsigurns = trms
    # number of terms for each signaler
    numsigterms = trms
    # runs
    # number of basic actions is the numrecterms
    numrecterms = trms
    
    plays = pls
    
    # cumulative success
    cumsuc = 0
    measuc = 0
    
    measure_suc = 10000
    
    step_suc = np.zeros(4)
    
    # number of times that learning dynamic alternates
    iterations = iters
    alternations = iterations*2
#     alt_list = list(range(0, plays+(plays // alternations), plays // alternations))
    alt_list = list(range(0, plays//alternations, plays // alternations))
    cap = plays // alternations
    
    numseeds = alternations*len(alt_list)+1
    
    
    
    # amount of reinforcement and punishment on each alternation
    reinarray = np.array([rein1, rein2])
    punarray = np.array([pun1, pun2])
    reinforcement = np.tile(reinarray, alternations+1)
    punishment = np.tile(punarray, alternations+1)
    
    
    
    
    sig_weights = np.ones([numsigurns, numsigterms])
    
    # number of receiver terms is going to be the same as the total number of statements that can be transmitted
    numrecurns = numsigterms
    
    rec_weights = np.ones([numrecurns, numrecterms])
    
    
    for it_00 in range(0, alternations):

        rein = reinforcement[it_00]
        punish = punishment[it_00]

        for it_01 in alt_list:
            
            it_pass = it_00*cap+it_01
            
            
            nature_list = rng.integers(0, numsigurns, cap)
            floats = rng.random((2, cap))
            picks_list = (floats[0]).copy()
            draws_list = (floats[1]).copy()
    
            step_suc, measuc, cumsuc, sig_weights, rec_weights = gen_sig_tile(n, it_pass, cap, step_suc, measuc, cumsuc, sig_weights, rec_weights, trms, rein, punish, iters, pls, nature_list, picks_list, draws_list)
            
            
    
    
    
    cumsuc = cumsuc/plays
    measuc = measuc/measure_suc
    
    final00 = [cumsuc, measuc, step_suc[0], step_suc[1], step_suc[2], step_suc[3]]
    
    
    return final00




@numba.jit
def gen_sig_tile(n, it_pass, cap, step_suc, measuc, cumsuc, sig_weights, rec_weights, trms, rein, punish, iters, pls, nature_list, picks_list, draws_list):


    #need to make sure a different seed is used each time we run the function



#     # number of signalers
#     numsig = 1
    # number of urns for each signaler
    numsigurns = trms
    # number of terms for each signaler
    numsigterms = trms
    # runs
    # number of basic actions is the numrecterms
    numrecterms = trms
    # plays per run
    plays = pls
    # number of times that learning dynamic alternates
    
    
    measure_suc = 10000
    
    steps = 10**6
    
    it_03 = 0






    for it_02 in range(it_pass, it_pass+cap):




        # determine a state of nature at random with equal probability
        nature = nature_list[it_03]


        sigweight_play = sig_weights[nature]

        #****ADD****
        sigweight_total = np.sum(sigweight_play)
        scumsum = np.cumsum(sigweight_play)
        
        xsum = np.zeros(len(sigweight_play), dtype=numba.int64)
        
        picknum = picks_list[it_03]*scumsum[-1]
        
        xsum[scumsum<picknum] = 1
        pick = np.sum(xsum)

# Leftover from b4 ADD
#         sigweight_norm[sigweight_norm < picks_list[it_03]] = 5
#         sigweight_norm[sigweight_norm != 5] = 0
#         sigweight_norm[sigweight_norm == 5] = 1


#         pick = math.floor(np.sum(sigweight_norm))

        recweight_play = rec_weights[pick]

        recweight_total = np.sum(recweight_play)
        rcumsum = np.cumsum(recweight_play)


        ysum = np.zeros(len(recweight_play), dtype=numba.int64)
        
        drawnum = draws_list[it_03]*rcumsum[-1]
        
        ysum[rcumsum<drawnum] = 1
        draw = np.sum(ysum)


        
# leftover from before ADD
#         recweight_norm[recweight_norm < draws_list[it_03]] = 5
#         recweight_norm[recweight_norm != 5] = 0
#         recweight_norm[recweight_norm == 5] = 1


#         draw = math.floor(np.sum(recweight_norm))

        if draw == nature:

            cumsuc = cumsuc + 1

            sigweight_play[pick] = sigweight_play[pick] + rein
            recweight_play[draw] = recweight_play[draw] + rein

            if it_02 > (plays - measure_suc):
                measuc = measuc + 1
            elif (plays -steps -measure_suc) < it_02 < (plays -steps):
                step_suc[3] = step_suc[3]+1
            elif (plays -(2*steps) -measure_suc) < it_02 < (plays -(2*steps)):
                step_suc[2] = step_suc[2]+1
            elif (plays -(3*steps) -measure_suc) < it_02 < (plays -(3*steps)):
                step_suc[1] = step_suc[1]+1
            elif (plays -(4*steps) -measure_suc) < it_02 < (plays -(4*steps)):
                step_suc[0] = step_suc[0]+1

        else:
            sigweight_play[pick] = sigweight_play[pick] + punish
            recweight_play[draw] = recweight_play[draw] + punish


        sigweight_play[sigweight_play < 1] = 1
        recweight_play[recweight_play < 1] = 1

        sig_weights[nature] = sigweight_play
        rec_weights[pick] = recweight_play
        
        
        it_03 = it_03+1
        

    
    
    return step_suc, measuc, cumsuc, sig_weights, rec_weights

