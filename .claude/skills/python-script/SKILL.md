---
name: python-script
description: When writing Python scripts, always include author information header.
---

When writing a Python script, always include the following header at the top of the file:

1. **Author information block**: Include a docstring or comment block with:
   - Author: Zehao Lu
   - Email: luzehao1998@gmail.com
   - Date created
   - Brief description of the script

2. **Use this format**:

```python
#!/usr/bin/env python3
"""
Author: Zehao Lu
Email: luzehao1998@gmail.com
Date: [YYYY-MM-DD]
Description: [Brief description of what this script does]
"""
```

3. **Additional guidelines**:
   - Include `#!/usr/bin/env python3` shebang for executable scripts
   - Add type hints where appropriate
   - Include a `if __name__ == "__main__":` block for scripts meant to be run directly

# Example

A script to calculate fibonacci numbers:

```python
#!/usr/bin/env python3
"""
Author: Zehao Lu
Email: luzehao1998@gmail.com
Date: 2026-01-21
Description: Calculate and print fibonacci numbers up to a given limit.
"""

def fibonacci(n: int) -> list[int]:
    """Generate fibonacci sequence up to n numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib


if __name__ == "__main__":
    result = fibonacci(10)
    print(f"Fibonacci sequence: {result}")
```
