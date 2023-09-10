# When mounted under "/app" routes call "app/lib" for static content instead of "/lib"
from typing import Optional, Mapping

from starlette.requests import Request

from ml_display.config import templates, main_menu
from ml_display.models import AlertInfo

base_context = {"base_path_reset": "../"}


def page_response(
    template_path: str,
    request: Request,
    context=None,
    status_code: int = 200,
    headers: Optional[Mapping[str, str]] = None,
    media_type: Optional[str] = None,
):
    if context is None:
        context = {}
    return templates.TemplateResponse(
        template_path,
        {**base_context, "request": request, "main_menu": main_menu, **context},
        status_code=status_code,
        headers=headers,
        media_type=media_type,
    )


def info_banner(request: Request, alert: AlertInfo):
    return templates.TemplateResponse(
        "components/info_banner.html",
        {
            **base_context,
            "request": request,
            "alert": alert,
        },
    )
