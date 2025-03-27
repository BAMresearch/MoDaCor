
from datetime import datetime, timezone
from typing import Callable, Optional
from attrs import define, field, validators as v


@define
class ModuleExecution:
    """A call of a processing step"""
    method: Callable = field(validator=v.is_callable())
    args: list = field(factory=list, validator=v.instance_of(list))
    kwargs: dict = field(factory=dict, validator=v.instance_of(dict))
    note: Optional[str] = field(default=None, validator=v.optional(v.instance_of(str)))
    start_time: Optional[datetime] = field(default=None)  # built-in profiling.... sort of. Will this do?
    stop_time: Optional[datetime] = field(default=None)  # built-in profiling.... sort of. Will this do?
   
    def start(self):
        self.start_time = datetime.now(tz=timezone.utc)
    
    def stop(self):
        self.stop_time = datetime.now(tz=timezone.utc)
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.stop_time:
            return (self.stop_time - self.start_time).total_seconds()
        return None
    
    def apply(self):
        self.start()
        result = self.method(*self.args, **self.kwargs)
        self.stop()
        return result