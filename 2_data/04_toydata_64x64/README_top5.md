The three files X64_top5.npy, Y64_top5.npy and ID64_top5.txt together form the dataset used for comparing different feature extraction methods.

- X64_top5.npy is a numpy array of shape (19037, 64, 64, 1) consisting of 19037 images each of which have 64x64 pixels with the NDVI channel/band precomputed.

- Y64_top5.npy is a numpy array of shape (19037, 5) where:
	- Y[:, 0] is the proportion (of area) of full crop loss.
	- Y[:, 1] is the proportion (of area) of partial crop loss.
	- Y[:, 2] is the loss category 4D that is derived from Y[:, 0] and Y[:, 1]. 
		- 1: only full loss (Y[:, 0] > 0 & Y[:, 1] == 0)
		- 2: only partial loss (Y[:, 0] == 0 & Y[:, 1] > 0)
		- 3: both full and partial loss (Y[:, 0] > 0 & Y[:, 1] > 0)
		- 4: no loss (Y[:, 0] == 0 & Y[:, 1] == 0)
	- Y[:, 3] is the loss category 2D that is derived from Y[:, 0] and Y[:, 1]. 
		- 0: no loss (Y[:, 0] == 0 & Y[:, 1] == 0)
		- 1: some loss (Y[:, 0] > 0 | Y[:, 1] > 0)
	- Y[:, 4] is the plant category. 
		- 0: Rehuohra
		- 1: Kaura
		- 2: Mallasohra
		- 3: Kevätvehnä
		- 4: Kevätrypsi
			
- ID64_top5.txt consists of the field parcel name/ids

NOTE: The order of rows of each of the three files are in corresponding order. That is, for a given i, the field parcel ID[i]'s attributes are in Y[i] and its image in X[i].
