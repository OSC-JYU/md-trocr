# Makefile for MD-trocr

# Variables
IMAGE_NAME = md-trocr
IMAGE_NAME_CPU = md-trocr-cpu
TAG = latest
CONTAINER_NAME = md-trocr-container
CONTAINER_NAME_CPU = md-trocr-cpu-container
PORT = 9012

# GPU version targets
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

.PHONY: run
run:
	docker run --gpus all -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):9012 \
		$(IMAGE_NAME):$(TAG)

.PHONY: run-interactive
run-interactive:
	docker run --gpus all -it --rm \
		--name $(CONTAINER_NAME) \
		-p $(PORT):9012 \
		$(IMAGE_NAME):$(TAG)

# CPU version targets
.PHONY: build-cpu
build-cpu:
	docker build -f Dockerfile.cpu -t $(IMAGE_NAME_CPU):$(TAG) .

.PHONY: run-cpu
run-cpu:
	docker run -d \
		--name $(CONTAINER_NAME_CPU) \
		-p $(PORT):9012 \
		$(IMAGE_NAME_CPU):$(TAG)

.PHONY: run-cpu-interactive
run-cpu-interactive:
	docker run -it --rm \
		--name $(CONTAINER_NAME_CPU) \
		-p $(PORT):9012 \
		$(IMAGE_NAME_CPU):$(TAG)

# Common targets
.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker stop $(CONTAINER_NAME_CPU) || true

.PHONY: remove
remove: stop
	docker rm $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME_CPU) || true

.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME) || docker logs -f $(CONTAINER_NAME_CPU)

.PHONY: shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash || docker exec -it $(CONTAINER_NAME_CPU) /bin/bash

.PHONY: clean
clean: remove
	docker rmi $(IMAGE_NAME):$(TAG) || true
	docker rmi $(IMAGE_NAME_CPU):$(TAG) || true

.PHONY: rebuild
rebuild: clean build

.PHONY: rebuild-cpu
rebuild-cpu: clean build-cpu

.PHONY: help
help:
	@echo "MD-trocr Makefile commands:"
	@echo ""
	@echo "GPU version:"
	@echo "  make build              - Build the GPU Docker image"
	@echo "  make run                - Run the GPU container in detached mode"
	@echo "  make run-interactive    - Run the GPU container in interactive mode"
	@echo ""
	@echo "CPU version:"
	@echo "  make build-cpu          - Build the CPU Docker image"
	@echo "  make run-cpu            - Run the CPU container in detached mode"
	@echo "  make run-cpu-interactive - Run the CPU container in interactive mode"
	@echo ""
	@echo "Common:"
	@echo "  make stop               - Stop running containers"
	@echo "  make remove             - Remove containers"
	@echo "  make logs               - View container logs"
	@echo "  make shell              - Access container shell"
	@echo "  make clean              - Remove containers and images"
	@echo "  make rebuild            - Clean and rebuild GPU image"
	@echo "  make rebuild-cpu        - Clean and rebuild CPU image"
	@echo "  make help               - Show this help message"
