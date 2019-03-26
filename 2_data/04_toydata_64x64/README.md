The three files X64.npy, Y64.npy and ID64.txt together form the dataset used for comparing different feature extraction methods. 
---X64.npy is a numpy array of shape (7477, 13, 64, 64) consistig of 7477 images each of which is 64x64 pixels with 13 channels/bands.
---Y64.npy is a numpy array of shape (7477, 4) where
	- Y[0] is the area of the field parcel. 
	- Y[1] is the proportion (of area) of full crop loss.
	- Y[2] is the proportion of partial crop loss.
	- Y[3] is the class label that is derived from Y[1] and Y[2]. 
		-- 1: only full loss (Y[:, 1] > 0 & Y[:, 2] == 0)
		-- 2: only partial loss (Y[:, 1] == 0 & Y[:, 2] > 0)
		-- 3: both full and partial loss (Y[:, 1] > 0 & Y[:, 2] > 0)
		-- 4: no loss (Y[:, 1] == 0 & Y[:, 2] == 0)
---ID64.txt consists of the field parcel name/ids

NOTE: The order of rows of each of the three files are in corresponding order. That is, for a given i, the field parcel ID[i]'s attributes are in Y[i] and its image in X[i].
