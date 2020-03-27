

rna:
	python rna_experiment.py --models SGIN lasso group_lasso --sampling balance
	python rna_experiment.py --models SGIN  --sampling all
	# Below are addition experiments for Algorithm 2 and SGD, which are not important for understanding SGIN
	python rna_experiment.py --models sgd theory  --sampling balance
	python rna_experiment.py --models sgd theory  --sampling all



