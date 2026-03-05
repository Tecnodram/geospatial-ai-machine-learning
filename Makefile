.PHONY: all train_fast train_full batch clean

all:
	python src/run_all.py --config config.yml

train_fast:
	python src/train_pipeline.py --config config.yml --dev

train_full:
	python src/train_pipeline.py --config config.yml

batch:
	python src/batch_blends.py --config config.yml

clean:
	rmdir /S /Q cache 2>nul || exit 0
	rmdir /S /Q submissions_batch 2>nul || exit 0
	del /Q submission_V*.csv 2>nul || exit 0