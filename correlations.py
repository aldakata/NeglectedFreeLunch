import numpy as np
from scipy.stats import spearmanr

# from matplotlib.pyplot import plot, savefig

# Function to be called for different hardness metrics and subsets
def correlation_coefficient_subset(x,y, mask):
    return np.corrcoef(x[mask],y[mask])[0,1]

def plot_loss_estimateTimes(wid, label, aid):
    mask = np.logical_and(worker_ids==wid, assignments==aid)
    mask = np.logical_and(mask, labels == label)
    # plt.scatter(losses[mask], estimate_times[mask])

if __name__ == "__main__":
    fnames_imagenet = np.asarray([el.split(".")[0] for el in np.load("./data/imagenet_file_names.npy")])
    fnames_nfl = np.asarray([el.split(".")[0] for el in np.load("./data/neglected_free_lunch_file_names.npy")])
    interesection_fnames = np.isin(fnames_imagenet, fnames_nfl)
    loss = np.load("./data/losses.npy")
    # print(f"Imagenet length {fnames_imagenet.shape}, NFL length {fnames_nfl.shape}, Loss length {loss.shape}, Loss masked {loss[interesection_fnames].shape}")

    estimate_times=np.load('data/sample_estimate_times.npy')
    selected_counts=np.load('data/sample_selected_counts.npy')
    selecteds=np.load('data/sample_selecteds.npy')
    worker_ids=np.load('data/sample_worker_ids.npy')
    losses = np.load('data/losses.npy')
    losses_86 = np.load('data/losses_86.npy')
    mouse_record_times=np.load('data/sample_mouse_record_times.npy')
    hovered_record_times=np.load('data/sample_hovered_record_times.npy')
    labels = np.load('data/sample_labels.npy')
    assignments = np.load('data/sample_assignment_ids.npy')
    # print(estimate_times.shape,selected_counts.shape,selecteds.shape,worker_ids.shape,losses.shape,losses_86.shape,mouse_record_times.shape,hovered_record_times.shape)
    # print(assignments.shape, labels.shape)
    # print(estimate_times[interesection_fnames][:10], worker_ids[interesection_fnames][:10])
    # print(labels[interesection_fnames][:10], assignments[interesection_fnames][:10], np.unique(worker_ids[interesection_fnames][:10]))
    unq, cnt = np.unique(assignments[assignments!=""], return_counts=True)
    print(unq.shape, cnt)
    mask = assignments==unq[-1]
    wunq, wcnt =  np.unique(worker_ids[interesection_fnames], return_counts=True)

    print(f"{selecteds.sum()}")
    # print(np.arange(len(fnames_imagenet))[mask], labels[mask], selecteds[mask],  wunq[1:], wcnt[1:].sum())
    # print(losses[mask], np.corrcoef(losses, labels))

    correlations = {}
    sranks = {}
    wid = "01CA538B53"
    aid = "3WAKVUDHU68MHMIB5UFINMQZUK1U70"
    l = 737
    for worker in [wid]:#np.unique(worker_ids)[1:]:
        wmask  = worker_ids==worker
        smask = np.logical_and(wmask, selecteds)
        correlations[worker]={}
        sranks[worker]={}

        for l in [l]:#np.unique(labels[wmask]):
            lmask = labels==l
            lwmask = np.logical_and(wmask,lmask)
            correlations[worker][l]={}
            sranks[worker][l]={}
            
            for assignment in [aid]:#np.unique(assignments[lwmask]):
                amask = assignments==assignment
                mask = np.logical_and(lwmask, amask)
                
                if mask.sum()>0:
                    correlations[worker][l][assignment] = np.corrcoef(losses[mask], estimate_times[mask])[0,1]
                    sranks[worker][l][assignment] = spearmanr(losses[mask], estimate_times[mask])
                    # print(worker, l,  np.corrcoef(losses[mask], estimate_times[mask])[0,1])
                    # print(worker, l,  spearmanr(losses[mask], estimate_times[mask]))
                    # print('---')
                    print(losses[mask], estimate_times[mask])
    print(np.corrcoef(losses[mask], estimate_times[mask])[0,1])
    print(spearmanr(losses[mask], estimate_times[mask]))


    print(sranks[wid][l][aid])
    