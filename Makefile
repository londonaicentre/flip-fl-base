# Copyright (c) 2026 Guy's and St Thomas' NHS Foundation Trust & King's College London
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
DEBUG ?= false

# Docker compose commands
DOCKER_COMPOSE_CMD = docker compose -f deploy/compose.yml
DOCKER_COMPOSE_DEV_CMD = NET_NUMBER=$(NET_NUMBER) docker compose -f deploy/compose.dev.yml up --build --remove-orphans
DOCKER_COMPOSE_TEST_CMD = NET_NUMBER=$(NET_NUMBER) docker compose -f deploy/compose.test.yml up --build --remove-orphans

# Test commands for development
lint_command = uv run ruff check . --fix
test_coverage_command = uv run pytest -s -vv --cov=flip/ --cov-report=term-missing tests/unit/

#======================================#
#         FL Network Commands          #
#======================================#

nvflare-provision:
	@./scripts/provision-network.sh net-${NET_NUMBER}_project.yml $(NET_NUMBER)

nvflare-provision-2-nets:
	NET_NUMBER=1 $(MAKE) nvflare-provision
	NET_NUMBER=2 $(MAKE) nvflare-provision

nvflare-provision-stag:
	@./scripts/provision-network.sh net-${NET_NUMBER}_project_stag.yml $(NET_NUMBER) workspace-stag

upload-flare-kits-to-s3:
	@datestr=$$(date +%Y%m%d) && \
	echo "Uploading FLARE participant kits for network $(NET_NUMBER) to S3 with date string: $$datestr" && \
	aws s3 sync ./workspace-stag/net-1 s3://flipstag-aicentre/fl-flare-participant-kits/$$datestr/net-1 --delete --dryrun

nvflare-provision-additional-client:
	@./scripts/provision-additional-client.sh $(NET_NUMBER) $(FL_PORT)

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
#          Test Data Commands          #
#======================================#

download-test-data:
	@if [ ! -d ".test_data" ]; then \
		mkdir -p .test_data && \
		aws s3 sync s3://$(FLIP_BUCKET_NAME)/test-data/flip-base-application .test_data; \
	else \
		echo "Directory .test_data already exists, skipping download."; \
	fi

#======================================#
#         Integration Tests            #
#======================================#

MERGED_DIR ?= .test_runs/merged-job-dir

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

test-xrays-standard:
	@./scripts/merge-job-dirs.sh src/standard/app tutorials/image_classification/xray_classification/app_files "$(MERGED_DIR)"
	$(TEST_XRAYS_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test-spleen-standard:
	@./scripts/merge-job-dirs.sh src/standard/app tutorials/image_segmentation/3d_spleen_segmentation/app_files "$(MERGED_DIR)"
	$(TEST_SPLEEN_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test-spleen-evaluation:
	@./scripts/merge-job-dirs.sh src/evaluation/app tutorials/image_evaluation/3d_spleen_segmentation_evaluation/app_files "$(MERGED_DIR)"
	@cp -v .test_data/checkpoints/model.pt "$(MERGED_DIR)/custom/model.pt"
	$(TEST_SPLEEN_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test-spleen-diffusion:
	@./scripts/merge-job-dirs.sh src/diffusion_model/app tutorials/image_synthesis/latent_diffusion_model/app_files "$(MERGED_DIR)"
	$(TEST_SPLEEN_VARS) JOB_DIR="../$(MERGED_DIR)" $(DOCKER_COMPOSE_TEST_CMD) nvflare-simulator-test

test:
	@echo "Running integration tests with filtered output (showing only errors, warnings, and test results)..."
	@echo "============================== XRays Standard Test =============================="
	$(MAKE) test-xrays-standard 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED"
	@echo "============================== Spleen Standard Test =============================="
	$(MAKE) test-spleen-standard 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED"
	@echo "============================== Spleen Evaluation Test =============================="
	$(MAKE) test-spleen-evaluation 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED"
	@echo "============================== Spleen Diffusion Test =============================="
	$(MAKE) test-spleen-diffusion 2>&1 | grep -i -A5 -B5 "make\[1\]: Leaving\|exited with code\|ERROR\|FAILED"

unit-test:
	# run unit tests with test coverage and verbose output, without capturing stdout
	$(lint_command) && $(test_coverage_command)

.PHONY: nvflare-provision build up down clean up-net down-net build-net \
        download-test-data \
		test-xrays-standard test-spleen-standard test-spleen-evaluation test-spleen-diffusion test \
		unit-test
