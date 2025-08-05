# Architecture

This document provides a high-level overview of the software architecture of the GA Optimizer.

## Core Principles

The architecture is designed to be modular, extensible, and easy to maintain. It follows the principles of separation of concerns, with distinct layers for domain logic, application services, and infrastructure.

## Directory Structure Overview

-   **`ga_optimizer/src`**: The main source code for the library.
    -   **`core/`**: Contains the core abstractions and interfaces of the application, such as abstract base classes for optimizers, evaluators, and data providers. This layer defines the contracts that other layers must adhere to.
    -   **`domain/`**: This layer represents the business domain of the application. It includes the domain models (e.g., `FuelSpec`, `PipelineNode`), business logic (e.g., blending calculations), and domain-specific concepts related to optimization (e.g., objectives, constraints).
    -   **`algorithms/`**: This layer contains the implementations of the optimization algorithms. It is organized by algorithm type (e.g., `genetic`, `local`, `hybrid`).
    -   **`data/`**: This layer is responsible for data management. It includes data providers for fetching data from various sources (e.g., synthetic data, BigQuery) and repositories for abstracting data access.
    -   **`evaluation/`**: This layer contains the evaluation and fitness functions used by the optimization algorithms. It is separated from the algorithms themselves to allow for flexible combination of algorithms and evaluation criteria.
    -   **`initialization/`**: This layer provides different strategies for initializing the population of the genetic algorithm (e.g., random initialization, smart seeding).
    -   **`services/`**: This layer contains the application services that orchestrate the main use cases of the application, such as running an optimization, analyzing results, and generating reports.
    -   **`utils/`**: A collection of utility functions for various purposes, such as serialization, visualization, and logging.
    -   **`cli/`**: The command-line interface for the application.

-   **`tests/`**: Contains the test suite, with separate directories for unit and integration tests.
-   **`examples/`**: Contains example scripts that demonstrate how to use the library.
-   **`config/`**: Contains configuration files for different environments (e.g., development, production).
-   **`requirements/`**: Contains the Python dependencies for the project.
-   **`docs/`**: Contains the project documentation.
