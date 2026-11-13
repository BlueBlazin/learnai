from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route


async def stream(request):
    return PlainTextResponse("not implemented", status_code=501)


app = Starlette(routes=[Route("/stream", stream)])
