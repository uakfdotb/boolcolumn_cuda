Installation:

	$ cd boolcolumn
	$ python setup.py install

Then look at example.py, e.g.:

	# mats must be a batch_size x nrows x ncols matrix
	import boolcolumn_cuda
	output = boolcolumn_cuda.forward(mats)
	# output will be batch_size x ncols x ncols
	# output[batch, i, j] == 1 implies mats[batch, k, i] >= mats[batch, k, j] for all k
