"""
Compatibility shim applied before bluesky_gym is imported.

The AI4REALNET bluesky-gym fork declares `requires-python = ">=3.10"` in its
pyproject.toml, but bluesky_gym/envs/common/screen_dummy.py does

    from warnings import deprecated

and `warnings.deprecated` only exists from Python 3.13 (PEP 702). On 3.10 to 3.12 the
import of any env module therefore fails with

    ImportError: cannot import name 'deprecated' from 'warnings'

which makes the whole fork, and so the AI4REALNET environment, unimportable on the
Python versions it claims to support. This has been reported upstream.

Setting the attribute on the already-imported `warnings` module makes the fork's
`from warnings import deprecated` resolve, without touching the fork's source. On 3.13
and later nothing is changed. The shim is a no-op once the fork is fixed, so it can be
removed then rather than needing to be maintained.

Import this module before bluesky_gym:

    from _bluesky_compat import ensure_warnings_deprecated
    ensure_warnings_deprecated()
    import bluesky_gym
"""

import logging
import warnings

logger = logging.getLogger(__name__)


def ensure_warnings_deprecated():
    """Provide warnings.deprecated on Python < 3.13. Returns True if it was added."""
    if hasattr(warnings, "deprecated"):
        return False
    try:
        from typing_extensions import deprecated as _deprecated
    except ImportError:
        # typing_extensions is not guaranteed to be installed. The decorator only marks
        # the callable it wraps, so an identity decorator preserves behaviour; what the
        # fork loses is the deprecation warning, not functionality.
        def _deprecated(*_args, **_kwargs):
            def _identity(obj):
                return obj
            return _identity

    warnings.deprecated = _deprecated
    logger.debug(
        "added warnings.deprecated for the bluesky-gym fork (Python < 3.13); "
        "see this module's docstring"
    )
    return True
