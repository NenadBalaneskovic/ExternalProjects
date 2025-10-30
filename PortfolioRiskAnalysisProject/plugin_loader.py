import os
import importlib.util


def discover_plugins(plugin_dir="plugins"):
    if not os.path.exists(plugin_dir):
        return
    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py"):
            path = os.path.join(plugin_dir, filename)
            spec = importlib.util.spec_from_file_location(filename[:-3], path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"Plugin {filename} failed to load: {e}")
