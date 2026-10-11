"""
MacOffload node facades and processing helpers.
"""
try:
    from ._api_routes import register_macoffload_routes
    register_macoffload_routes()
except Exception:
    pass

