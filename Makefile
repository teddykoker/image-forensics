PHONY: download

SHELL=/bin/bash

download:
	mkdir -p data
	mkdir -p tmp
	
	# BBBC038 Kaggle 2018 Datascience Bowl
	
	# use provided training set for validation
	wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip -O tmp/bbbc038_valid.zip
	unzip "tmp/bbbc038_valid.zip" -d tmp/bbbc038_valid/
	mkdir -p data/valid/bbbc038/
	find tmp/bbbc038_valid/ -path "*/images/*.png" -exec mv {} data/valid/bbbc038/ \;
	
	# use stage 1 test set for testing
	wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip -O tmp/bbbc038_test.zip
	unzip "tmp/bbbc038_test.zip" -d tmp/bbbc038_test/
	mkdir -p data/test/bbbc038/
	find tmp/bbbc038_test/ -path "*/images/*.png" -exec mv {} data/test/bbbc038/ \; 	
	
	# use large test set for training
	wget https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip -O tmp/bbbc038_train.zip
	unzip "tmp/bbbc038_train.zip" -d tmp/bbbc038_train/
	mkdir -p data/train/bbbc038/
	find tmp/bbbc038_train/ -path "*/images/*.png" -exec mv {} data/train/bbbc038/ \;
	

	rm -rf tmp
