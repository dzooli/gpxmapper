# GitHub Copilot Context

This is the default Copilot prompt for this project.

## Tooling

- uv for package management using dependency groups
- ruff for code formatting
- Pytest for test automation
- Pytest-cov for test coverage generation
- Typer for command line interface
- SonarCloud for code quality analysis
- PyInstaller for executable creation
- Github Actions for continuous integration
- Github Releases for release management
- mkdocs for end-user documentation generation
- Git for source code versioning

## Project structure

- Usinng a virtual environment is mandatory. It is in the .venv directory.
- Source files in the src directory, packaging ready
- The package name and properties in the pyproject.toml file
- Ready for packaging
- Test files in the tests directory
- docs/devdocs for development documentation and notes
- docs/userdocs for end-user documentation
- docs/assets for the documentation assets
- docs/build for the end-user documentation build output
- docs/build/index.md for the documentation homepage
- pytest.ini for pytest executor configuration
- tests/conftest.py for fixtures and other Pytest configuration

## Project Description

This project is created to generate a short video file using a recorded GPS track in a GPX file.
It provides a command line interface using the typer Python package.

### Features

- fixed title text
- scrolling caption from a text file
- timed captions from a csv file where timestamps and captions are stored
- reverse geolocation using an external Nominatim server
- TrueType font usage for text rendering
- Windows executable creation by Github Actions

## Guidelines

### Agentic workflow

- Use step-by-step planning
- Use a knowledge graph if available to store information about the project
- Use a knowledge graph if available to store found troubleshooting steps

### Software design principles

As a professional software developer who is also experienced in test automation you should follow
these rules:

- Use software design patterns
- Apply SOLID desing principles as:
    - Single Responsibility Principle: A class should have only one reason to change, meaning it should have only one
      job.
    - Open/Closed Principle: Software entities should be open for extension but closed for modification.
    - Liskov Substitution Principle: Subtypes must be substitutable for their base types without altering the
      correctness of the program.
    - Interface Segregation Principle: Clients should not be forced to depend on interfaces they do not use.
    - Dependency Inversion Principle: High-level modules should not depend on low-level modules; both should depend on
      abstractions.
- Apply DRY principle
- Apply KISS principle
- Apply YAGNI principle
- Keep the method and function cognitive complexity below 15
- Keep method and function length below 30 lines
- Use slotted dataclasses when applicable
- Use generators and iterators when applicable
- Use effective Python data structures and algorithms

