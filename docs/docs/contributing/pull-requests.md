# Pull Requests

How to submit changes to WakeBuilder.

---

## Before You Start

1. Check [existing issues](https://github.com/XableTech/WakeBuilder/issues)
2. Open an issue to discuss major changes
3. Fork the repository

---

## Branch Naming

Use descriptive branch names with prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/add-noise-augmentation` |
| `fix/` | Bug fixes | `fix/websocket-connection-error` |
| `docs/` | Documentation | `docs/update-installation-guide` |
| `refactor/` | Code refactoring | `refactor/training-pipeline` |
| `test/` | Test additions | `test/add-trainer-tests` |

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `style` | Formatting (no code change) |
| `refactor` | Code restructuring |
| `test` | Adding tests |
| `chore` | Maintenance |

### Examples

```
feat(training): add pitch shift augmentation

Implement pitch shifting in data augmentation pipeline to improve
model robustness across different voice pitches.

Closes #123
```

```
fix(backend): handle WebSocket disconnection gracefully

Add proper error handling for WebSocket disconnections during
real-time testing to prevent server crashes.

Fixes #456
```

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes
# ...

# Format and lint
black src/ tests/
ruff check src/ tests/ --fix

# Run tests
pytest
```

### 2. Commit Your Changes

```bash
git add .
git commit -m "feat(module): describe your change"
```

### 3. Push to Your Fork

```bash
git push origin feature/my-feature
```

### 4. Open Pull Request

On GitHub:

1. Click "New Pull Request"
2. Select your branch
3. Fill in the template
4. Submit

---

## PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran and their results

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings generated
```

---

## Review Process

### What Reviewers Look For

- **Correctness**: Does it work?
- **Style**: Does it follow guidelines?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it documented?
- **Breaking changes**: Does it break existing code?

### Responding to Feedback

- Address all comments
- Explain your decisions
- Push additional commits
- Re-request review when ready

### After Merge

- Delete your branch
- Pull latest main
- Celebrate! 🎉

---

## Tips for Getting PRs Merged

!!! success "Do"

    - Keep PRs focused and small
    - Write clear descriptions
    - Include tests
    - Update documentation
    - Respond to feedback quickly

!!! failure "Avoid"

    - Large PRs with many unrelated changes
    - Skip tests or documentation
    - Ignore review comments
    - Force push without notice

---

## First-Time Contributors

New to open source? Here's how to start:

1. Look for issues labeled `good first issue`
2. Comment on the issue to claim it
3. Ask questions if you're stuck
4. Start small and build up

Welcome to the community! 🎙️
