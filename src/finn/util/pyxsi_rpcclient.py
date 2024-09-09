import xmlrpc.client


class PyXSIRPCProxy(object):
    def __init__(self, wrapped=None):
        if wrapped is None:
            wrapped = xmlrpc.client.ServerProxy("http://localhost:8000")
        self.wrapped = wrapped

    def __getattr__(self, name):
        attr = getattr(self.wrapped, name)
        return type(self)(attr)

    def __call__(self, *args, **kw):
        return self.wrapped(*args, **kw)
