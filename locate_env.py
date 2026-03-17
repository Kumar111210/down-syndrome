import os
from pathlib import Path

folder = Path.cwd().resolve()
print("You are currently in folder:", folder)

env_path = folder / ".env"

print("\nLooking for .env here →", env_path)
print("File exists?", env_path.exists())

if env_path.exists():
    try:
        content = env_path.read_text(encoding="utf-8").strip()
        print("\n=== ACTUAL CONTENT OF .env ===")
        print(content)
        print("=== END ===")
    except Exception as e:
        print("Cannot read file:", e)
else:
    print("No .env file found in this folder!")