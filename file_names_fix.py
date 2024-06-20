import numpy as np

if __name__ == "__main__":
    fnames_nfl = np.asarray([el.split(".")[0] for el in np.load("./data/neglected_free_lunch_file_names.npy")])
    fnames_imagenet = np.asarray([el.split(".")[0] for el in np.load("./data/imagenet_file_names.npy")])
   
    print("Neglected Free Lunch file names: ", fnames_nfl)
    print("ImageNet file names: ", fnames_imagenet)
    missing_fnames = np.isin(fnames_imagenet, fnames_nfl, invert=True)
    print("Missing file names: ", fnames_imagenet[missing_fnames])
    np.save("./data/missing_file_names_mask.npy", fnames_imagenet[missing_fnames])
    print("Missing file names mask: ", missing_fnames)
    np.save("./data/missing_file_names_mask.npy", missing_fnames)
    print("Missing file names: ", missing_fnames.sum())