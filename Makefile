USER_ID=$(shell id -u)
GROUP_ID=$(shell id -g)

DATA_DIR=$(shell pwd):/app/data
RAW_DATA_DIR=$(DATA_DIR)/raw
PROCESSED_DATA_DIR=$(DATA_DIR)/processed
SPLIT_DATA_DIR=$(DATA_DIR)/split

build:
		docker build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) -t pdiow-python .

dvc_repro:
	docker run -it --rm -v $(shell pwd):/app pdiow-python dvc repro

dvc_status:
	docker run -it --rm -v $(shell pwd):/app pdiow-python dvc status

dvc_track_raw:
	docker run -it --rm -v $(shell pwd):/app pdiow-python dvc add $(RAW_DATA_DIR)

dvc_track_processed:
	docker run -it --rm -v $(shell pwd):/app pdiow-python dvc add $(PROCESSED_DATA_DIR)

dvc_track_split:
	docker run -it --rm -v $(shell pwd):/app pdiow-python dvc add $(SPLIT_DATA_DIR)

dvc_experiment:
	docker run -it --rm -v $(shell pwd):/app -v ~/.gitconfig:/etc/gitconfig -u $(USER_ID):$(GROUP_ID) pdiow-python dvc exp run -S train.model=uniform

dvc_experiment_show:
	docker run -it --rm -v $(shell pwd):/app -v ~/.gitconfig:/etc/gitconfig -u $(USER_ID):$(GROUP_ID) pdiow-python dvc exp show

dvc_status:
	docker run -it --rm -v $(shell pwd):/app pdiow-python dvc status