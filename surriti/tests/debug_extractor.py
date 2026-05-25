"""Debug what the DummyExtractor produces."""
import sys
sys.path.insert(0, "/home/squire/projects/surriti/surriti")

from surriti.extractor import DummyExtractor

ext = DummyExtractor()

# Episode 1
result1 = ext.extract("Alice works at Acme Corp.")
print("=== Episode 1: 'Alice works at Acme Corp.' ===")
print(f"Entities: {result1.entities}")
print(f"Facts:")
for f in result1.facts:
    print(f"  {f.subject} {f.operation} {f.predicate} {f.object} replaces={f.replaces}")

# Episode 2
result2 = ext.extract("Alice no longer works at Acme Corp; Alice moved to Globex.")
print("\n=== Episode 2: 'Alice no longer works at Acme Corp; Alice moved to Globex.' ===")
print(f"Entities: {result2.entities}")
print(f"Facts:")
for f in result2.facts:
    print(f"  {f.subject} {f.operation} {f.predicate} {f.object} replaces={f.replaces}")