# CLAUDE.md

## Project Overview

flip-fl-base is the NVIDIA FLARE federated learning base library for the FLIP (Federated Learning Interoperability
Platform) ecosystem. It provides a pip-installable Python package (`flip-utils`), reusable NVFLARE components, Docker
services for FL networks, and tutorial applications for medical imaging. Developed by the AI Centre for Value Based
Healthcare — a collaboration between Guy's and St Thomas' NHS Foundation Trust and King's College London.

**License**: Apache 2.0 — all source files must include the copyright header.

## Repository Structure

```bash
flip-fl-base/
├── flip/                       # Core pip-installable Python package
│   ├── constants/              # FlipConstants, enums, PTConstants
│   ├── core/                   # FLIPBase, Standard implementations, factory
│   ├── nvflare/                # NVFLARE-specific components
│   │   ├── controllers/        # Workflow controllers (ScatterAndGather, CrossSiteModelEval)
│   │   ├── executors/          # RUN_TRAINER, RUN_VALIDATOR, RUN_EVALUATOR
│   │   └── components/         # Event handlers, persistors, privacy filters, locators
│   └── utils/                  # General utilities, model weight helpers
├── src/                        # Application job type templates
│   ├── standard/app/           # Standard federated training (FedAvg)
│   ├── evaluation/app/         # Distributed evaluation jobs
│   ├── diffusion_model/app/    # Two-stage VAE + diffusion training
│   └── fed_opt/app/            # Custom federated optimization
├── fl_services/                # Docker services for FL networks
│   ├── fl-base/                # Base NVFLARE image with dependencies
│   ├── fl-server/              # NVFLARE server container
│   ├── fl-client/              # NVFLARE client container
│   └── fl-api-base/            # FastAPI admin API service (own pyproject.toml)
├── tutorials/                  # Example FL applications
│   ├── image_classification/   # X-ray classification
│   ├── image_segmentation/     # 3D spleen segmentation
│   ├── image_synthesis/        # Latent diffusion model
│   ├── image_evaluation/       # Evaluation examples
│   └── testing/                # Test utilities
├── tests/                      # Unit tests (mirrors flip/ structure)
│   └── unit/
├── docs/                       # Sphinx documentation
├── deploy/                     # Docker Compose files
│   ├── compose.yml             # Production compose
│   ├── compose.dev.yml         # Development compose
│   └── compose.test.yml        # Integration test compose
├── scripts/                    # Provisioning and helper scripts
├── .github/workflows/          # CI/CD workflows
├── Makefile                    # Build, test, and deployment automation
├── pyproject.toml              # Project metadata and dependencies (UV/Hatch)
├── net-1_project.yml           # NVFLARE provisioning config (network 1)
├── net-2_project.yml           # NVFLARE provisioning config (network 2)
└── net-1_project_stag.yml      # Staging provisioning config
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.12+ |
| FL Framework | NVIDIA FLARE 2.7.1 |
| ML / Medical Imaging | MONAI, PyTorch, PyDICOM, NiBabel (optional `full` extras) |
| Data Validation | Pydantic 2.11+, Pydantic-Settings |
| Data Handling | Pandas |
| Cloud Storage | AWS S3 (Boto3) |
| Package Management | UV (`uv sync`, `uv add`), Hatch (build backend) |
| Testing | pytest, pytest-cov |
| Linters / Formatters | Ruff (linter + formatter), MyPy (type checking) |
| Containers | Docker, Docker Compose |
| Container Registry | GitHub Container Registry (GHCR) |
| CI/CD | GitHub Actions |
| Documentation | Sphinx, ReadTheDocs |

## Common Commands

### Installation

```bash
uv sync                    # Install dependencies (dev + core)
uv sync --all-extras       # Install with full ML dependencies
pip install .              # Alternative: pip install
uv build                   # Build distributable package
```

### Testing

```bash
make unit-test             # Lint (ruff) + unit tests with coverage
uv run pytest -s -vv       # Run pytest directly with verbose output
```

### Integration Tests (require Docker + test data)

```bash
make download-test-data          # Download test data from HuggingFace
make test                        # Run all integration tests
make test-xrays-standard         # X-ray classification test
make test-spleen-standard        # 3D spleen segmentation test
make test-spleen-evaluation      # Evaluation job test
make test-spleen-diffusion       # Diffusion model test
make test-spleen-fedopt          # Federated optimization test
```

### Linting & Type Checking

```bash
uv run ruff check . --fix  # Lint with auto-fix
uv run mypy .              # Static type checking
```

### Docker / FL Network

```bash
make build NET_NUMBER=1    # Build Docker images for a network
make up NET_NUMBER=1       # Start FL network (server, clients, API)
make down NET_NUMBER=1     # Stop FL network
make clean                 # Remove containers and images
```

### NVFLARE Provisioning

```bash
make nvflare-provision NET_NUMBER=1              # Provision network
make nvflare-provision-2-nets                    # Provision both networks
make nvflare-provision-stag                      # Provision for staging
make nvflare-provision-additional-client         # Add new client
```

### Documentation

```bash
make docs                  # Build Sphinx HTML documentation
make docs-clean            # Clean build artifacts
```

## Workflow Requirements

### Always Use Make Commands

When a `Makefile` target exists for the task at hand, **always use it** instead of running raw commands. Make targets
encapsulate the correct flags, environment setup, and command sequences. Key rules:

- Use `make unit-test` rather than invoking `uv run ruff` and `uv run pytest` separately — it runs lint + tests in the
  correct order.
- Use `make build` / `make up` / `make down` instead of raw `docker compose` commands.
- Check the `Makefile` for available targets before writing manual commands.

### Always Verify Changes

After making any code changes, **always run verification before committing**:

1. **Run the test suite**:

   ```bash
   make unit-test   # Runs ruff lint + pytest with coverage
   ```

2. **For FL API changes**, also run:

   ```bash
   cd fl_services/fl-api-base && make test   # API-specific tests
   ```

3. **Fix all failures** before committing — do not commit code that fails linting or tests.

### Check If Documentation Needs Updating

After making changes, **always evaluate whether documentation needs to be updated**:

| Change Type | Documentation to Review |
| --- | --- |
| New NVFLARE components | `docs/`, service-level `README.md` |
| New job type templates | `src/` README files, `docs/` |
| Changed environment variables | `.env.development`, `.env.production.template` |
| New dependencies | `pyproject.toml`, `README.md` |
| Changed Docker/deployment config | `deploy/` compose files, `README.md` |
| New Make targets | `README.md`, this file (`CLAUDE.md`) |
| New tutorials | `tutorials/` README files, `docs/` |
| FL API changes | `fl_services/fl-api-base/README.md` |

When in doubt, update the docs. Outdated documentation is worse than no documentation.

## Code Style & Conventions

### Python

- **Line length**: 120 characters
- **Linter/Formatter**: Ruff (`select = ['I', 'F', 'E', 'W', 'PT']`)
- **Type checker**: MyPy (Python 3.12, `ignore_missing_imports = true`)
- **Docstrings**: Google style guide
- **Naming**: snake_case for functions, variables, modules
- **Imports**: Alphabetically sorted (enforced by Ruff `I` rule)
- **Source layout**: `flip/` for the core package, `src/` for job templates
- **Test files**: `test_*.py` in `tests/unit/` (mirrors `flip/` package structure)

### Architecture

- **Factory pattern**: `FLIP()` factory returns appropriate implementation (Dev or Prod)
- **Abstract base class**: `FLIPBase` with concrete `FLIPStandard`, `FLIPDev` implementations
- **Plugin-based executors**: Trainers, validators, evaluators registered via NVFLARE config
- **Configuration as code**: NVFLARE JSON configs + Pydantic settings

### General

- All files must include the Apache 2.0 copyright header
- Commits must be signed off (DCO): `git commit -s`
- PRs target the `develop` branch
- Branch naming: `[ticket_id]-[task_name]` (e.g., `19-ci-pipeline-setup`)

## Environment Setup

1. Install Python 3.12+ and UV
2. Clone the repo and run `uv sync`
3. Copy env template: `cp .env.development .env.development.local` (if customizing)
4. For integration tests: `make download-test-data`
5. For Docker services: ensure Docker and Docker Compose are installed

### Key Environment Variables

- `NET_NUMBER` — FL network number (default: `1`)
- `FL_PORT` — FL client port (default: `8002`)
- `DEBUG` — enable debug mode (`true`/`false`)
- `LOCAL_DEV` — build flag for local development
- `DEV_IMAGES_DIR` — path to test image data
- `DEV_DATAFRAME` — path to test dataframe CSV

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

| Workflow | Purpose |
| --- | --- |
| `unit-tests.yml` | Ruff lint + unit tests + MyPy + Codecov (on every push) |
| `fl_api.yml` | FL API-specific tests |
| `docker_build_fl.yml` | Build and push Docker images to GHCR (main/develop) |
| `push-app-to-s3-*.yml` | Sync FL app templates to S3 (dev/stag/prod) |
| `push-pr-to-s3.yml` | Upload PR-specific app bundles to S3 |
| `cleanup-pr-s3.yml` | Clean up PR S3 artifacts on merge/close |
| `release-pypi.yml` | Publish `flip-utils` package to PyPI |
| `check-package-metadata.yml` | Validate package metadata |
| `check-version-bump.yml` | Ensure version is bumped on PRs |
| `check-required-files.yml` | Verify required files are present |
| `pr_acceptance_criteria.yml` | Validate PR requirements |
| `pr-release-notes-preview.yml` | Preview release notes |

## Docker Services

The FL network consists of four Docker services:

| Service | Description |
| --- | --- |
| `fl-base` | Base NVFLARE image with core dependencies |
| `fl-server` | NVFLARE server (federation coordinator) |
| `fl-client` | NVFLARE client (participant worker) |
| `fl-api-base` | FastAPI service for FL run control and status |

Compose files in `deploy/`:
- `compose.yml` — production stack (pulls GHCR images)
- `compose.dev.yml` — development stack (builds from source)
- `compose.test.yml` — integration test stack (NVFLARE simulator)

## Security Rules

- Never commit secrets or credentials
- Never bypass TLS certificate validation
- Use environment variables for configuration — do not hardcode values in Dockerfiles or compose files
- AWS OIDC authentication for CI/CD (no long-lived keys)

## Important: Rules for Adding or Modifying Code

Follow these rules when adding new code or modifying existing code:

1. Follow existing code style and conventions
2. Add or update tests in `tests/` to cover new functionality
3. Run formatters, linters, and tests before committing (`make unit-test`)
4. Update documentation in `README.md` or `docs/` as needed
5. Commit changes with clear messages referencing related issues or features. NEVER co-sign commits or PRs — all commits
   must be signed off by the human author alone (`git commit -s`), as they are the sole responsible for the content and
   quality of the code.
6. Ensure that any new dependencies are added to `pyproject.toml` and documented
7. Use the SOLID principles for code organization and design, ensuring that new code is modular, reusable, and
   maintainable
8. Measure code coverage and aim for high coverage on new code paths

## Related Repositories

| Repository | Purpose |
| --- | --- |
| [FLIP](https://github.com/londonaicentre/FLIP) | Main platform mono-repo (Central Hub, Trust services, UI) |
| [flip-fl-base](https://github.com/londonaicentre/flip-fl-base) | NVIDIA FLARE FL base library (this repo) |
| [flip-fl-base-flower](https://github.com/londonaicentre/flip-fl-base-flower) | Flower FL base library |
