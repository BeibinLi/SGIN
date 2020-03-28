

rna:
	python rna_experiment.py --models SGIN lasso group_lasso --sampling balance
	python rna_experiment.py --models SGIN  --sampling all
	# Below are addition experiments for Algorithm 2 and SGD, which are not important for understanding SGIN
	python rna_experiment.py --models sgd theory  --sampling balance
	python rna_experiment.py --models sgd theory  --sampling all




asd:
	python et_asd_classification.py --models SGIN lasso group_lasso
	# Additional optimizer experiments
	python et_asd_classification.py --models theory nn sgd


regression:
	python et_regression.py --models SGIN lasso --task ados
	python et_regression.py --models SGIN lasso --task iq
	python et_regression.py --models SGIN lasso --task srs
	python et_regression.py --models SGIN lasso --task vineland
