# Contributing to DevMentor AI

Thank you for your interest in contributing to DevMentor AI! This document provides guidelines and instructions for contributing.

## Ways to Contribute

### 1. Code Contributions
- **Data Pipeline**: Improve data collection, preprocessing, or validation
- **Training**: Optimize training loops, add new techniques
- **Evaluation**: Create new benchmarks or improve existing ones
- **Deployment**: Enhance inference performance, add deployment options
- **Documentation**: Improve guides, add examples, fix typos

### 2. Non-Code Contributions
- **Bug Reports**: Report issues you encounter
- **Feature Requests**: Suggest new capabilities
- **Documentation**: Improve clarity and completeness
- **Testing**: Help test new features
- **Community**: Answer questions, help other users

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/devmentor-ai.git
cd devmentor-ai

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/devmentor-ai.git
```

### 2. Set Up Development Environment

```bash
# Run setup
bash scripts/setup.sh

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/bug-description
```

## Development Workflow

### 1. Make Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings and comments
- Update documentation as needed

### 2. Test Your Changes

```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_training.py

# Run with coverage
pytest --cov=src tests/
```

### 3. Code Quality Checks

```bash
# Format code with black
black src/

# Check with flake8
flake8 src/

# Type checking with mypy
mypy src/

# Or run all checks
pre-commit run --all-files
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: description of what you did"
```

**Commit Message Guidelines**:
- Use present tense ("Add feature" not "Added feature")
- Be concise but descriptive
- Reference issues if applicable (#123)

Examples:
```
✓ Add support for CodeLlama tokenizer
✓ Fix memory leak in data preprocessing
✓ Update documentation for deployment
✓ Improve performance of attention computation
✗ update stuff
✗ fix bug
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Pull Request Guidelines

### PR Title Format
- `[Feature] Add support for X`
- `[Fix] Resolve issue with Y`
- `[Docs] Update guide for Z`
- `[Performance] Optimize W`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No breaking changes (or documented if necessary)
```

## Code Style Guide

### Python Style

We follow PEP 8 with some modifications:

```python
# Good: Clear, documented, typed
def process_code_sample(
    code: str,
    language: str,
    quality_threshold: float = 0.5
) -> Optional[CodeSample]:
    """
    Process a code sample with quality filtering.

    Args:
        code: Source code content
        language: Programming language
        quality_threshold: Minimum quality score (0.0-1.0)

    Returns:
        Processed CodeSample or None if below threshold
    """
    if not code.strip():
        return None

    quality = compute_quality(code, language)

    if quality < quality_threshold:
        logger.warning(f"Low quality sample: {quality:.2f}")
        return None

    return CodeSample(
        content=code,
        language=language,
        quality_score=quality
    )


# Bad: No types, no docs, unclear
def process(c, l, t=0.5):
    if not c:
        return None
    q = compute(c, l)
    if q < t:
        return None
    return CodeSample(c, l, q)
```

### Key Guidelines

1. **Type Hints**: Use type hints for all function parameters and return values
2. **Docstrings**: Use Google-style docstrings
3. **Line Length**: Max 100 characters (not 80)
4. **Imports**: Use absolute imports, group by stdlib/third-party/local
5. **Naming**:
   - `snake_case` for functions and variables
   - `PascalCase` for classes
   - `UPPER_CASE` for constants

## Testing Guidelines

### Writing Tests

```python
# tests/test_data_engine.py
import pytest
from src.data_engine.processors.preprocess import Deduplicator


class TestDeduplicator:
    """Test suite for Deduplicator class"""

    @pytest.fixture
    def deduplicator(self):
        """Create deduplicator instance"""
        return Deduplicator(similarity_threshold=0.9)

    def test_exact_duplicate_detection(self, deduplicator):
        """Test detection of exact duplicates"""
        code = "def hello():\n    print('Hello')"

        # First occurrence should not be duplicate
        assert not deduplicator.is_duplicate(code)

        # Second occurrence should be duplicate
        assert deduplicator.is_duplicate(code)

    def test_near_duplicate_detection(self, deduplicator):
        """Test detection of near-duplicates"""
        code1 = "def hello():\n    print('Hello')"
        code2 = "def hello():\n    print('World')"  # Different string

        assert not deduplicator.is_duplicate(code1)
        # Should still detect as near-duplicate (same structure)
        assert deduplicator.is_duplicate(code2)
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests for complex workflows

## Documentation Guidelines

### Code Documentation

```python
class DataCollector:
    """
    Collects training data from various sources.

    This class manages the collection of code samples from GitHub,
    GitLab, and other repositories while respecting rate limits
    and license requirements.

    Attributes:
        max_repos: Maximum number of repositories to collect
        min_stars: Minimum star count for repositories
        approved_licenses: Set of allowed license types

    Example:
        >>> collector = DataCollector(max_repos=100)
        >>> async with collector:
        ...     data = await collector.collect_language_data("Python")
    """
```

### Guide Documentation

- Use clear, step-by-step instructions
- Include code examples
- Add troubleshooting sections
- Keep language accessible

## Review Process

### What Reviewers Look For

1. **Correctness**: Does it work as intended?
2. **Code Quality**: Is it readable and maintainable?
3. **Tests**: Are changes properly tested?
4. **Documentation**: Are changes documented?
5. **Performance**: Are there performance implications?
6. **Security**: Are there security concerns?

### Responding to Reviews

- Be respectful and professional
- Address all feedback
- Ask questions if unclear
- Make requested changes promptly

## Areas Needing Contribution

### High Priority

- [ ] Expand benchmark coverage (more coding tasks)
- [ ] Optimize data preprocessing pipeline
- [ ] Add support for more programming languages
- [ ] Improve documentation with more examples
- [ ] Create video tutorials

### Medium Priority

- [ ] Add support for different tokenizers
- [ ] Implement model distillation
- [ ] Create Jupyter notebook examples
- [ ] Add performance profiling tools
- [ ] Improve error messages

### Future Enhancements

- [ ] Multi-modal support (code + diagrams)
- [ ] IDE integrations (VS Code, PyCharm)
- [ ] Web-based playground
- [ ] Mobile apps
- [ ] Browser extensions

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Significant contributors may become maintainers.

## Code of Conduct

### Our Standards

- **Be Respectful**: Treat everyone with respect
- **Be Constructive**: Provide helpful feedback
- **Be Inclusive**: Welcome diverse perspectives
- **Be Professional**: Maintain professional conduct

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Reporting

Report violations to: conduct@devmentor-ai.example.com

## Questions?

- **Technical Questions**: Open a GitHub Discussion
- **Bug Reports**: Create a GitHub Issue
- **Security Issues**: Email security@devmentor-ai.example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DevMentor AI! Together we're building the future of developer assistance. 🚀
