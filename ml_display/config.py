from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from starlette.templating import Jinja2Templates

app_path = Path(__file__).parent
to_generate_path = app_path / "templates" / "to_pregenerate"
static_path = app_path / "static"
generated_path = app_path.parent / "generated"

templates = Jinja2Templates(directory=app_path / "templates")


@dataclass
class NavItem:
    name: str
    target_slug: str
    children: list[NavItem]


main_menu = [
    NavItem("Home", "", []),
    NavItem("Other", "other.html", []),
    # NavItem("Books", "app/books", []),
]
# protected_routes = ["/app/books"]
