#!/bin/bash



# Crawl ntc-scv data from https://github.com/congnghia0609/ntc-scv
git clone https://github.com/congnghia0609/ntc-scv.git
unzip ./ntc-scv/data/data_train.zip -d data
unzip ./ntc-scv/data/data_test.zip -d data
rm -rf ntc-scv

# Install labraries from requirements.txt
pip install -r requirements.txt