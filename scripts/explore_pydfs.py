"""
Explore pydfs-lineup-optimizer structure
"""

from pydfs_lineup_optimizer import get_optimizer, Site, Sport
import inspect

# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)

# Explore attributes
print("Optimizer attributes:")
print("=" * 60)
for attr in dir(optimizer):
    if not attr.startswith('_'):
        try:
            value = getattr(optimizer, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except:
            print(f"{attr}: <unable to access>")

print("\n\nOptimizer methods:")
print("=" * 60)
for attr in dir(optimizer):
    if not attr.startswith('_'):
        try:
            value = getattr(optimizer, attr)
            if callable(value):
                sig = inspect.signature(value) if hasattr(inspect, 'signature') else ''
                print(f"{attr}{sig}")
        except:
            print(f"{attr}()")

print("\n\nSettings attributes:")
print("=" * 60)
for attr in dir(optimizer.settings):
    if not attr.startswith('_'):
        try:
            value = getattr(optimizer.settings, attr)
            if not callable(value):
                print(f"settings.{attr}: {value}")
        except:
            print(f"settings.{attr}: <unable to access>")

print("\n\nPlayer Pool attributes:")
print("=" * 60)
for attr in dir(optimizer.player_pool):
    if not attr.startswith('_'):
        try:
            value = getattr(optimizer.player_pool, attr)
            if not callable(value):
                print(f"player_pool.{attr}: {value}")
        except:
            print(f"player_pool.{attr}: <unable to access>")