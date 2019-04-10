The three files X64.npy, Y64.npy and ID64.txt together form the dataset used for comparing different feature extraction methods.

- X64.npy is a numpy array of shape (18951, 64, 64, 1) consisting of 18951 images each of which have 64x64 pixels with the NDVI channel/band precomputed.

- Y64.npy is a numpy array of shape (18951, 5) where:
	- Y[:, 0] is the proportion (of area) of full crop loss.
	- Y[:, 1] is the proportion (of area) of partial crop loss.
	- Y[:, 2] is 4-class class labels derived from Y[:, 0] and Y[:, 1]. 
		- 0: only full loss (Y[:, 0] > 0 & Y[:, 1] == 0)
		- 1: only partial loss (Y[:, 0] == 0 & Y[:, 1] > 0)
		- 2: both full and partial loss (Y[:, 0] > 0 & Y[:, 1] > 0)
		- 3: no loss (Y[:, 0] == 0 & Y[:, 1] == 0)
	- Y[:, 3] is a different 2-class labelling of data (for experimental purpose) based on the 4-class labels above
		- 0: no loss (formed by class label 3 in Y[:, 2])
		- 1: some loss (formed by combining class labels 0, 1 and 2 in Y[:, 2])
	- Y[:, 4] is the plant category. 
		- 0: Rehuohra
		- 1: Kaura
		- 2: Mallasohra
		- 3: Kevätvehnä
		- 4: Kevätrypsi
			
- ID64.txt consists of the field parcel name/ids

NOTE: The order of rows of each of the three files are in corresponding order. That is, for a given i, the field parcel ID[i]'s attributes are in Y[i] and its image in X[i].

- Finally, the folder mavi_shape_files contains the shape files of the fields. The ordering of recorders in the shape files may not match the files above.

