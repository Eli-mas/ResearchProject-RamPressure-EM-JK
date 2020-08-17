#import os
from pathlib import Path
from queue import Queue, Empty
from types import ModuleType

from asy_io.asy_paths import MAIN_PROGRAM_DIRECTORY
from __init__ import modules as mods

def list_modules():
	q = Queue()
	
	for m in mods:
		q.put(m)
	
	modules = []
	
	while True:
		try:
			m = q.get_nowait()
			modules.append(m.__name__)
			
			for name in m.__all__:
				print(name)
				obj = getattr(m,name)
				if isinstance(obj, ModuleType):
					if MAIN_PROGRAM_DIRECTORY in Path(obj.__file__).parents:
						q.put(obj)
		
		except AttributeError as e:
			print(f'AttributeError on {m.__name__}:',repr(e))
		
		except Empty:
			break
	
	return modules

all_modules = list_modules()
print(all_modules)