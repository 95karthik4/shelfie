#!/usr/bin/env python
"""Django's command-line utility.

This file's location is load-bearing. Python puts the directory containing the
script it runs at the front of sys.path, so keeping manage.py in backend/
makes backend/ the import root -- which is exactly what the existing
vision/, vlm/ and matching/ packages already assume (they are imported as
top-level packages, not as backend.*). That is why there is no
backend/__init__.py and no sys.path manipulation anywhere in this project.
"""

import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
