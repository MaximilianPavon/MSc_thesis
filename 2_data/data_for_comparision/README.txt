The dataset is used to compare various methods for learning low-dim representation of the images. The three files X64.npy, Y64.npy and ID64.txt together makup the dataset.

-- X64.npy: A numpy array of shape (7495, 13, 64, 64) consisting of 7495 images where each image 64 pixles in height and width with 13 channels. 
-- Y64.npy: A numpy array of shape (nimages, 4) 
    ---- column Y[0] is the total area of the field.
    ---- column Y[1] is the proprtion of the field with full crop loss.
    ---- column Y[2] is the proportion of the field partial crop loss
    ---- column Y[3] is class of the images (1 - full loss only, 2 - partial loss only, 3 - full and partial, 4 - no loss).
-- ID64.txt: A text file containing the field name/ID of each image in the same order as X64.npy. 

NOTE: The elements of each files are in corresponding order. That is, for a given index i, the image X64[i] has field name ID64[i] and labels Y64[i]
          
