"""Test-only ProcessStep access tracing; no patches survive fixture teardown."""

from contextvars import ContextVar
from functools import wraps

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, matches_processing_pattern
from modacor.dataclasses.processing_data import ProcessingData

TRACE_ENABLED = ContextVar("contract_trace_enabled", default=False)
_ACTIVE_TRACE = ContextVar("active_contract_trace", default=None)


class AccessTrace:
    """Attribute bundle accesses by identity, retaining objects to prevent id reuse."""

    def __init__(self, data):
        self.data = data
        self.bundles = {}
        self.accesses = set()
        self.refresh()

    def refresh(self):
        self.bundles = {}
        if self.data is not None:
            for key, bundle in dict.items(self.data):
                self.bundles.setdefault(id(bundle), (bundle, set()))[1].add(key)

    def record(self, bundle, key, kind):
        entry = self.bundles.get(id(bundle))
        if entry is not None:
            for processing_key in entry[1]:
                self.accesses.add((f"{processing_key}.{key}", kind))

    def check(self, step):
        contract = step.dependency_contract()
        patterns = contract.processing_reads | contract.processing_writes
        for path, kind in sorted(self.accesses):
            if not matches_processing_pattern(path, patterns):
                raise AssertionError(f"{type(step).__name__}: undeclared {kind} access {path!r}; contract={contract!r}")


def _wrap_step(method, *, execute=False):
    if getattr(method, "_contract_trace_wrapped", False):
        return method

    @wraps(method)
    def traced(self, *args, **kwargs):
        if not TRACE_ENABLED.get() or _ACTIVE_TRACE.get() is not None:
            return method(self, *args, **kwargs)
        data = (args[0] if args else kwargs["data"]) if execute else self.processing_data
        trace = AccessTrace(data)
        token = _ACTIVE_TRACE.set(trace)
        try:
            result = method(self, *args, **kwargs)
        finally:
            _ACTIVE_TRACE.reset(token)
        # Reached only after a successful return; checking must not itself be traced.
        trace.check(self)
        return result

    traced._contract_trace_wrapped = True
    return traced


def _wrap_class(cls, monkeypatch):
    for name in ("calculate", "prepare_execution"):
        method = cls.__dict__.get(name)
        if method is not None:
            monkeypatch.setattr(cls, name, _wrap_step(method))
    for child in cls.__subclasses__():
        _wrap_class(child, monkeypatch)


def _wrap_bundle(monkeypatch, name, kind, *, whole=False):
    original = getattr(DataBundle, name)

    @wraps(original)
    def access(self, *args, **kwargs):
        trace = _ACTIVE_TRACE.get()
        if trace is not None:
            key = "*" if whole else (args[0] if args else kwargs["key"])
            trace.record(self, key, kind)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataBundle, name, access)


def _wrap_processing_insert(monkeypatch):
    original = ProcessingData.__setitem__

    @wraps(original)
    def insert(self, key, bundle):
        trace = _ACTIVE_TRACE.get()
        tracked = trace is not None and self is trace.data
        previous = dict.get(self, key)
        existed = dict.__contains__(self, key)
        original(self, key, bundle)
        if tracked:
            trace.refresh()
            if previous is not bundle:
                if existed:
                    trace.accesses.add((f"{key}.*", "write"))
                else:
                    for basedata_key in dict.keys(bundle):
                        trace.accesses.add((f"{key}.{basedata_key}", "write"))

    monkeypatch.setattr(ProcessingData, "__setitem__", insert)


def _wrap_processing_removal(monkeypatch, name):
    original = getattr(ProcessingData, name)

    @wraps(original)
    def remove(self, key, *args, **kwargs):
        trace = _ACTIVE_TRACE.get()
        tracked = trace is not None and self is trace.data
        if tracked:
            trace.accesses.add((f"{key}.*", "write"))
        result = original(self, key, *args, **kwargs)
        if tracked:
            trace.refresh()
        return result

    monkeypatch.setattr(ProcessingData, name, remove)


def install_contract_trace(monkeypatch):
    """Patch existing and future steps, using the caller's teardown-owned MonkeyPatch."""
    original_hook = ProcessStep.__dict__.get("__init_subclass__")

    def init_subclass(cls, **kwargs):
        if original_hook is None:
            super(ProcessStep, cls).__init_subclass__(**kwargs)
        else:
            original_hook.__get__(None, cls)(**kwargs)
        _wrap_class(cls, monkeypatch)

    monkeypatch.setattr(ProcessStep, "__init_subclass__", classmethod(init_subclass), raising=False)
    monkeypatch.setattr(ProcessStep, "execute", _wrap_step(ProcessStep.execute, execute=True))
    _wrap_class(ProcessStep, monkeypatch)
    for name in ("__getitem__", "get", "__contains__"):
        _wrap_bundle(monkeypatch, name, "read")
    for name in ("__setitem__", "__delitem__", "pop", "setdefault"):
        _wrap_bundle(monkeypatch, name, "write")
    for name in ("keys", "items", "values", "__iter__", "copy"):
        _wrap_bundle(monkeypatch, name, "read", whole=True)
    # DataBundle.update validates then delegates each key to __setitem__, including kwargs.
    _wrap_processing_insert(monkeypatch)
    for name in ("__delitem__", "pop"):
        _wrap_processing_removal(monkeypatch, name)
