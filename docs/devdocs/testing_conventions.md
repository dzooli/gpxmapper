

# Recommended Directory for Python Package Test Suites

The recommended directory structure for Python package test suites follows these conventions:

## Primary Options

1. **`tests/` directory at the root level** - This is the most common and recommended approach:
   ```
   my_package/
   ├── my_package/
   │   ├── __init__.py
   │   └── module.py
   ├── tests/
   │   ├── __init__.py
   │   ├── test_module.py
   │   └── ...
   ├── pyproject.toml
   └── ...
   ```

2. **`test/` directory (singular)** - Less common but still acceptable:
   ```
   my_package/
   ├── my_package/
   │   ├── __init__.py
   │   └── module.py
   ├── test/
   │   ├── __init__.py
   │   ├── test_module.py
   │   └── ...
   ├── pyproject.toml
   └── ...
   ```

## Alternative Approaches

3. **Tests inside the package** - For smaller projects:
   ```
   my_package/
   ├── my_package/
   │   ├── __init__.py
   │   ├── module.py
   │   └── tests/
   │       ├── __init__.py
   │       └── test_module.py
   ├── pyproject.toml
   └── ...
   ```

4. **Tests alongside modules** - Less common, but used in some projects:
   ```
   my_package/
   ├── my_package/
   │   ├── __init__.py
   │   ├── module.py
   │   └── test_module.py
   ├── pyproject.toml
   └── ...
   ```

## Best Practices

- The most widely adopted convention is the top-level `tests/` directory (option 1)
- Test files should be named with a `test_` prefix (e.g., `test_module.py`)
- Include an `__init__.py` file in your test directory to make it a proper package
- Popular test frameworks like pytest will automatically discover tests in these conventional locations

This structure keeps tests separate from the main package code while maintaining a clear organization that test runners can easily discover.

## Most Popular Directory Structure for Python Package Test Suites

Based on Python packaging conventions, **`my_package/tests`** (tests at the root level) is significantly more popular than **`my_package/my_package/tests`** (tests inside the package).

The root-level tests directory (`my_package/tests/`) is:
- The most widely adopted convention in the Python ecosystem
- Recommended by most Python packaging guides
- Used by the majority of popular Python packages
- Preferred by test frameworks like pytest for automatic test discovery

While placing tests inside the package (`my_package/my_package/tests/`) is a valid alternative approach, it's generally only recommended for smaller projects or specific use cases where tests need to access package-private resources.

The root-level tests directory provides better separation between production code and test code, making it the preferred choice for most Python projects.