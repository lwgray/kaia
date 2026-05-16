"""
Shared test fixtures for Kaia tests.
"""

from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_marcus_root(tmp_path: Path) -> Path:
    """Create a temporary Marcus-like directory structure"""
    marcus = tmp_path / "marcus"
    marcus.mkdir()

    # Create sample structure
    (marcus / "src").mkdir()
    (marcus / "src" / "core").mkdir()
    (marcus / "docs").mkdir()
    (marcus / "tests").mkdir()

    return marcus


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create a sample Python file for testing"""
    file = tmp_path / "sample.py"
    file.write_text('''"""Sample module for testing"""

class TaskCoordinator:
    """Coordinates task assignment"""

    def request_next_task(self, agent_id: str) -> dict:
        """
        Request next available task for agent.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier

        Returns
        -------
        dict
            Task data or empty dict if none available
        """
        return {}

def standalone_function() -> None:
    """A standalone function"""
    pass
''')
    return file


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing"""
    file = tmp_path / "README.md"
    file.write_text("""# Marcus Documentation

## Overview

Marcus is a multi-agent coordination framework.

## Architecture

### Task Coordination

Tasks are coordinated via a board-mediated pattern.

```python
coordinator = TaskCoordinator()
task = coordinator.request_next_task(agent_id)
```

## Error Handling

Use Marcus Error Framework for all errors.
""")
    return file
