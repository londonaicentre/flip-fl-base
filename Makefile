# Copyright (c) Guy's and St Thomas' NHS Foundation Trust & King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Load environment variables from .env.development file
ifneq ("$(wildcard .env.development)","")
include .env.development
export $(shell sed 's/=.*//' .env.development)
endif

# Default values
NET_NUMBER ?= 1
FL_PORT ?= 8002
ADMIN_PORT ?= 8003
DEBUG ?= false
LOG_LEVEL ?= DEBUG
MERGED_DIR ?= .test_runs/merged-job-dir

# Docker compose commands
DOCKER_COMPOSE_CMD = docker compose -f deploy/compose.yml
DOCKER_COMPOSE_DEV_CMD = NET_NUMBER=$(NET_NUMBER) docker compose -f deploy/compose.dev.yml up --build --remove-orphans
DOCKER_COMPOSE_TEST_CMD = NET_NUMBER=$(NET_NUMBER) docker compose -f deploy/compose.test.yml up --build --remove-orphans
# Create a unique temporary merge directory and merge two folders into it
# Usage:
#   $(call merge_dirs,sourceA,sourceB)
merge_dirs = \
	$(info 🗂️  Merging $(1) and $(2) into $(MERGED_DIR)) \
	$(shell rm -rf "$(MERGED_DIR)" 2>/dev/null || sudo rm -rf "$(MERGED_DIR)") \
	$(shell mkdir -p "$(MERGED_DIR)/custom" 2>/dev/null || sudo mkdir -p "$(MERGED_DIR)/custom" && sudo chown -R $(USER):$(USER) "$(MERGED_DIR)") \
	$(shell cp -r "$(1)"/* "$(MERGED_DIR)/" 2>/dev/null || true) \
	$(shell cp -r "$(2)"/* "$(MERGED_DIR)/custom/" 2>/dev/null || true) \
	$(info ✅ Done. Output: $(MERGED_DIR))

#======================================#
#         FL Network Commands          #
#======================================#

nvflare-provision:
	@./scripts/provision-network.sh $(NET_NUMBER) $(FL_PORT) $(ADMIN_PORT) $(DEBUG) $(LOG_LEVEL)

nvflare-provision-2-nets:
	NET_NUMBER=1 FL_PORT=8002 ADMIN_PORT=8003 DEBUG=$(DEBUG) LOG_LEVEL=$(LOG_LEVEL) $(MAKE) nvflare-provision 
	NET_NUMBER=2 FL_PORT=8002 ADMIN_PORT=8003 DEBUG=$(DEBUG) LOG_LEVEL=$(LOG_LEVEL) $(MAKE) nvflare-provision

nvflare-provision-additional-client:
	@./scripts/provision-additional-client.sh $(NET_NUMBER) $(FL_PORT) $(ADMIN_PORT)

build:
	@echo "Building Docker images for network $(NET_NUMBER) with LOCAL_DEV=$(LOCAL_DEV)"
	$(DOCKER_COMPOSE_CMD) --profile build-only build --build-arg LOCAL_DEV=$(LOCAL_DEV)

up:
	$(DOCKER_COMPOSE_CMD) -p net-$(NET_NUMBER) up --build --remove-orphans

down:
	$(DOCKER_COMPOSE_CMD) -p net-$(NET_NUMBER) down

clean:
	$(DOCKER_COMPOSE_CMD) down --rmi local && docker system prune -f

# Backwards compatibility aliases
up-net: up
down-net: down
build-net: build

#======================================#
#       Development Commands           #
#======================================#

run-container:
	@echo "Starting the application container..."
	@echo "  DEV_IMAGES_DIR=$(DEV_IMAGES_DIR)"
	@echo "  DEV_DATAFRAME=$(DEV_DATAFRAME)"
	@./scripts/check-dev-paths.sh ./deploy "$(DEV_IMAGES_DIR)" "$(DEV_DATAFRAME)"
	@sleep 3
	$(DOCKER_COMPOSE_DEV_CMD) nvflare-simulator-dev

#======================================#
#          Test Data Commands          #
#======================================#

download-xrays-data:
	@if [ ! -d ".test_data/xrays" ]; then \
		mkdir -p .test_data/xrays && \
		aws s3 sync s3://$(FLIP_BUCKET_NAME)/test-data/flip-base-application/xrays .test_data/xrays; \
	else \
		echo "Directory .test_data/xrays already exists, skipping download."; \
	fi

download-spleen-data:
	@if [ ! -d ".test_data/spleen" ]; then \
		mkdir -p .test_data/spleen && \
		aws s3 sync s3://$(FLIP_BUCKET_NAME)/test-data/flip-base-application/spleen .test_data/spleen; \
	else \
		echo "Directory .test_data/spleen already exists, skipping download."; \
	fi

download-checkpoints:
	@if [ ! -d ".test_data/checkpoints" ]; then \
		mkdir -p .test_data/checkpoints && \
		aws s3 sync s3://$(FLIP_BUCKET_NAME)/test-data/flip-base-application/checkpoints .test_data/checkpoints; \
	else \
		echo "Directory .test_data/checkpoints already exists, skipping download."; \
	fi

#======================================#
#         Integration Tests            #
#======================================#

# Test environment variables for xrays tests
TEST_XRAYS_VARS = \
	DEV_IMAGES_DIR=../.test_data/xrays/images \
	DEV_DATAFRAME=../.test_data/xrays/sample_get_dataframe_response.csv \
	RUNS_DIR=../.test_runs/xrays

# Test environment variables for spleen tests
TEST_SPLEEN_VARS = \
	DEV_IMAGES_DIR=../.test_data/spleen/accession-resources \
	DEV_DATAFRAME=../.test_data/spleen/sample_get_dataframe_response.csv \
	RUNS_DIR=../.test_runs/spleen

test-xrays-standard: download-xrays-data
	@./scripts/merge-job-dirs.sh src/standard/app tutorials/image_classification/xray_classification/app_files "$(MERGED_DIR)"
	$(TEST_XRAYS_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test-spleen-standard: download-spleen-data
	@./scripts/merge-job-dirs.sh src/standard/app tutorials/image_segmentation/3d_spleen_segmentation/app_files "$(MERGED_DIR)"
	$(TEST_SPLEEN_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test-spleen-evaluation: download-spleen-data download-checkpoints
	@./scripts/merge-job-dirs.sh src/evaluation/app tutorials/image_evaluation/3d_spleen_segmentation_evaluation/app_files "$(MERGED_DIR)"
	@cp -v .test_data/checkpoints/model.pt "$(MERGED_DIR)/custom/model.pt"
	$(TEST_SPLEEN_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test-spleen-diffusion: download-spleen-data
	@./scripts/merge-job-dirs.sh src/diffusion_model/app tutorials/image_synthesis/latent_diffusion_model/app_files "$(MERGED_DIR)"
	$(TEST_SPLEEN_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test:
	@echo "Running integration tests with filtered output (showing only errors, warnings, and test results)..."
	@echo "============================== XRays Standard Test =============================="
	$(MAKE) test-xrays-standard 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED\|WARNING"
	@echo "============================== Spleen Standard Test =============================="
	$(MAKE) test-spleen-standard 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED\|WARNING"
	@echo "============================== Spleen Evaluation Test =============================="
	$(MAKE) test-spleen-evaluation 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED\|WARNING"
	@echo "============================== Spleen Diffusion Test =============================="
	$(MAKE) test-spleen-diffusion 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED\|WARNING"

unit-test:
	# run unit tests with test coverage and verbose output, without capturing stdout
	uv run pytest -s -vv --cov=flip/ --cov-report=term-missing tests/unit/

#======================================#
#       Test App Management            #
#======================================#

SPLEEN_APP_FILES = config.json models.py trainer.py validator.py transforms.py

copy-spleen-app:
	cp -rv tutorials/image_segmentation/3d_spleen_segmentation/app_files/* src/standard/app/custom/

save-spleen-app:
	@for f in $(SPLEEN_APP_FILES); do \
		cp -fv src/standard/app/custom/$$f tutorials/image_segmentation/3d_spleen_segmentation/app_files/$$f; \
	done

.PHONY: nvflare-provision build up down clean up-net down-net build-net \
        run-container download-spleen-data download-checkpoints \
		test-xrays-standard test-spleen-standard test-spleen-evaluation test-spleen-diffusion test unit-test \
        copy-spleen-app save-spleen-app
