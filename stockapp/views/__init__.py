"""Views, one module per page. Re-exported so urls.py can keep using `views.<name>`."""
from .auth import register
from .dcf import dcf_view
from .frontier import efficient_frontier
from .portfolio import home

__all__ = ["home", "efficient_frontier", "dcf_view", "register"]
