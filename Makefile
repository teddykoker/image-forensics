PHONY: download

SHELL=/bin/bash

download:
	mkdir -p data
	mkdir -p tmp
	
	# mouse embreo	
	#wget http://celltracking.bio.nyu.edu/MouEmbTrkDtbCount.zip -O tmp/mouse.zip
	#unzip "tmp/mouse.zip" -d tmp/
	#mv tmp/MouEmbTrkDtbCount/CountEmbryos/Im/ 
	#mkdir -p data/train/mouse_emb
	#mkdir -p data/mouse/mouse_emb
	#mv tmp/MouEmbTrkDtbCount/CountEmbryos/Im/Ex{001..90}.png data/mouse/train/
	#mv tmp/MouEmbTrkDtbCount/CountEmbryos/Im/data/mouse/*.png data/mouse/test/
	
	# BBBC038 Kaggle 2018 Datascience Bowl
	
	# wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip -O tmp/bbbc038_train.zip
	# unzip "tmp/bbbc038_train.zip" -d tmp/bbbc038_train/
	# mkdir -p data/train/bbbc038/
	# find tmp/bbbc038_train/ -path "*/images/*.png" -exec mv {} data/train/bbbc038/ \;
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
