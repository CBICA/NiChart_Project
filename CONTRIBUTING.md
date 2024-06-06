# NiChart Contribution Guide

Welcome to the NiChart project! This guide outlines the necessary steps and principles for integrating new models, pipelines, and packages into the NiChart ecosystem. Adherence to these guidelines ensures consistency, usability, and maintainability across contributions.

## Contribution Principles

### 1. Language and Implementation

- **Language**: All contributions must be written in Python, C, R, or Matklab.
- **CLI Interface**: Provide a command-line interface with standard arguments such as `--version`, `--help`, etc. The CLI should be user-friendly and follow conventional syntax and semantics.

### 2. GitHub Repository Standards

To be considered for integration into the NiChart project, your GitHub repository must meet the following criteria:

#### Overview

- **Explanation**: Provide a clear and concise summary of the model, pipeline, or package.
- **Requirements**: List all necessary hardware (e.g., GPU) and software dependencies.

#### Repository Files

- **LICENSE**: Include an appropriate open-source license.
- **CONTRIBUTING.md**: Detail guidelines for contributing to the repository.
- **Documentation**: Comprehensive documentation explaining the functionality, usage, and implementation details.

#### Model Artifact

- **Distribution**: The model should be packaged as a distribution package, stored in Git LFS, or included as a small file within the repository.

#### README

- **Content**: A complete README file that includes:
  - Installation instructions
  - Usage examples
  - Example code
  - Results and outputs
- **References**: Include relevant publications or references.

#### Additional Files

- **Contribution Guide**: Specific guidelines for contributing to the repository.
- **License**: Ensure a clear and appropriate license is included.
- **Documentation**: Detailed documentation should be available, covering all aspects of the repository.

### 3. Packaging and Distribution

- **PyPI**: Publish the package on PyPI for easy installation and management.
- **DockerHub**: Create and publish a Docker container on DockerHub to facilitate deployment and integration.

### 4. Workflow Integration

- **Snakemake**: (Optional) Provide a Snakemake recipe file. This is particularly useful for internal use and enables running processes in batches and on high-performance computing (HPC) environments.

## Integration Process

1. **Preparation**: Ensure your repository adheres to the standards outlined above.
2. **Contact**: Reach out to the NiChart developer team for the integration process.
3. **Review**: The NiChart team will review your contribution for compliance and relevance.
4. **Approval**: Upon approval, your package will be included as a git submodule in the NiChart project.

---

Thank you for contributing to NiChart and enhancing neuroscience research! Your efforts help build a robust and comprehensive framework for the neuroimaging community.

For any questions or further assistance, please contact the NiChart developer team.

---

Feel free to suggest any improvements or additional requirements for this guide. Your feedback is valuable and helps us maintain a high standard for all contributions.
