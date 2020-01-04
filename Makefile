PHONY: download

SHELL=/bin/bash

download:
	mkdir -p data
	mkdir -p tmp
	wget http://celltracking.bio.nyu.edu/MouEmbTrkDtbCount.zip -O tmp/mouse.zip
	unzip "tmp/mouse.zip" -d tmp/
	mv tmp/MouEmbTrkDtbCount/CountEmbryos/Im/ data/mouse/
	rm -rf tmp
	mkdir -p data/mouse/train
	mkdir -p data/mouse/test
	mv data/mouse/Ex{001..90}.png data/mouse/train/
	mv data/mouse/*.png data/mouse/test/
